#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using Microsoft::WRL::ComPtr;
constexpr double PI=3.141592653589793, FS=48000, SPEED=343;
constexpr int CH=16, N=2048, BLOCK=8192, GW=51, GH=35;
using Complex=std::complex<double>;
void hr(HRESULT value, const char* label) {
    if (FAILED(value)) { char s[200]; sprintf_s(s,"%s (0x%08lX)",label,(unsigned long)value); throw std::runtime_error(s); }
}
struct Vec { double x,y,z; };
std::array<Vec,CH> positions() {
    const int layout[4][4]={{8,7,10,9},{6,5,12,11},{4,3,14,13},{2,1,16,15}};
    std::array<Vec,CH> p{};
    // Manual's microphone-side drawing faces the viewer; camera looks the opposite way.
    for(int r=0;r<4;r++) for(int c=0;c<4;c++) p[layout[r][c]-1]={(1.5-c)*.042,(1.5-r)*.042,0};
    return p;
}
void fft(std::vector<Complex>& a,bool inverse=false) {
    int n=(int)a.size();
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1) j^=b;j^=b;if(i<j)std::swap(a[i],a[j]);}
    for(int len=2;len<=n;len*=2){Complex step=std::polar(1.,(inverse?2:-2)*PI/len);
        for(int i=0;i<n;i+=len){Complex w=1.;for(int j=0;j<len/2;j++){auto u=a[i+j],v=a[i+j+len/2]*w;a[i+j]=u+v;a[i+j+len/2]=u-v;w*=step;}}}
    if(inverse)for(auto& v:a)v/=n;
}
struct Estimate { std::array<double,CH> db{}; std::vector<double> map; double x=0,y=0,level=-120,contrast=0; bool clipping=false; };
struct Pair { int a,b; std::vector<int> lag; std::vector<double> frac; };
class Beamformer {
    std::array<Vec,CH> pos=positions(); std::vector<Pair> pairs;
public:
    std::vector<Vec> rays;
    Beamformer(){
        for(int y=0;y<GH;y++)for(int x=0;x<GW;x++){
            double sx=(2.*x/(GW-1)-1)*2.75,sy=(1-2.*y/(GH-1))*2.0;
            double norm=sqrt(sx*sx+sy*sy+1);rays.push_back({sx/norm,sy/norm,1/norm});}
        for(int a=0;a<CH;a++)for(int b=a+1;b<CH;b++){
            Pair p{a,b};for(auto u:rays){double lag=-FS*((pos[a].x-pos[b].x)*u.x+(pos[a].y-pos[b].y)*u.y)/SPEED;
                int lower=(int)floor(lag);p.lag.push_back((lower+N)%N);p.frac.push_back(lag-lower);}pairs.push_back(p);}
    }
    Estimate process(const std::vector<float>& audio){
        Estimate out;out.map.assign(GW*GH,0);double energy=0;
        for(int c=0;c<CH;c++){double e=0;for(int t=0;t<BLOCK;t++){double v=audio[t*CH+c];e+=v*v;out.clipping|=fabs(v)>.99;}energy+=e;out.db[c]=10*log10(std::max(e/BLOCK,1e-16));}
        out.level=10*log10(std::max(energy/(BLOCK*CH),1e-16));
        std::vector<std::vector<Complex>> spectra(CH,std::vector<Complex>(N));
        std::vector<Complex> cross(N);
        for(int offset=0;offset+N<=BLOCK;offset+=N){
            for(int c=0;c<CH;c++){for(int t=0;t<N;t++)spectra[c][t]=audio[(offset+t)*CH+c]*(.5-.5*cos(2*PI*t/(N-1)));fft(spectra[c]);}
            for(auto& p:pairs){
                std::fill(cross.begin(),cross.end(),Complex{});
                for(int k=1;k<N/2;k++)if(k*FS/N>=700 && k*FS/N<=3200){
                    Complex v=spectra[p.a][k]*std::conj(spectra[p.b][k]);v/=std::max(std::abs(v),1e-12);cross[k]=v;cross[N-k]=std::conj(v);}
                fft(cross,true);
                for(int g=0;g<GW*GH;g++){int l=p.lag[g];double f=p.frac[g];out.map[g]+=(1-f)*cross[l].real()+f*cross[(l+1)%N].real();}
            }
        }
        double scale=(double)(BLOCK/N)*pairs.size();for(auto& v:out.map)v/=scale;
        auto best=std::max_element(out.map.begin(),out.map.end());int index=(int)(best-out.map.begin());
        double mean=std::accumulate(out.map.begin(),out.map.end(),0.)/out.map.size(),var=0;
        for(double v:out.map)var+=(v-mean)*(v-mean);var/=out.map.size();
        out.contrast=(*best-mean)/sqrt(std::max(var,1e-20));
        out.x=rays[index].x/rays[index].z;out.y=rays[index].y/rays[index].z;
        return out;
    }
};

// PortAudio ABI: use the already-installed ASIO DLL; no driver or Python changes.
struct PaDeviceInfo { int version; const char* name; int host,inputs,outputs; double lowIn,lowOut,highIn,highOut,rate; };
struct PaHostInfo { int version,type;const char* name;int count,input,output; };
struct PaParams { int device,channels; unsigned long format; double latency; void* info; };
class Audio {
    HMODULE dll=nullptr; void* stream=nullptr; bool initialized=false;
    int (__cdecl *initialize)();int (__cdecl *terminate)();int (__cdecl *count)();
    const PaDeviceInfo* (__cdecl *device)(int);const PaHostInfo* (__cdecl *host)(int);
    const char* (__cdecl *error)(int);
    int (__cdecl *open)(void**,const PaParams*,const PaParams*,double,unsigned long,unsigned long,void*,void*);
    int (__cdecl *start)(void*);int (__cdecl *read)(void*,void*,unsigned long);
    int (__cdecl *stop)(void*);int (__cdecl *close)(void*);
    template<class T> void load(T& f,const char* name){f=reinterpret_cast<T>(GetProcAddress(dll,name));if(!f)throw std::runtime_error(name);}
    void check(int rc){if(rc<0)throw std::runtime_error(error(rc));}
public:
    std::string name;int overflows=0;
    Audio(){
        wchar_t base[MAX_PATH];GetModuleFileNameW(nullptr,base,MAX_PATH);std::wstring path(base);path=path.substr(0,path.find_last_of(L"\\/"));
        path+=L"\\..\\ika-simulator\\.venv\\Lib\\site-packages\\_sounddevice_data\\portaudio-binaries\\libportaudio64bit-asio.dll";
        dll=LoadLibraryW(path.c_str());if(!dll)throw std::runtime_error("Cannot load existing PortAudio ASIO DLL");
        load(initialize,"Pa_Initialize");load(terminate,"Pa_Terminate");load(count,"Pa_GetDeviceCount");load(device,"Pa_GetDeviceInfo");load(host,"Pa_GetHostApiInfo");load(error,"Pa_GetErrorText");
        load(open,"Pa_OpenStream");load(start,"Pa_StartStream");load(read,"Pa_ReadStream");load(stop,"Pa_StopStream");load(close,"Pa_CloseStream");
        check(initialize());initialized=true;
        int selected=-1;
        for(int i=0;i<count();i++){auto d=device(i);if(d->inputs>=16 && std::string(d->name).find("miniDSP")!=std::string::npos && std::string(host(d->host)->name)=="ASIO"){selected=i;break;}}
        if(selected<0)throw std::runtime_error("16-channel miniDSP ASIO device not found. Close other audio apps and reconnect the array.");
        auto d=device(selected);name=d->name;PaParams params{selected,CH,1,d->highIn,nullptr};
        check(open(&stream,&params,nullptr,FS,1024,0,nullptr,nullptr));check(start(stream));
    }
    void capture(std::vector<float>& out){out.resize(BLOCK*CH);int rc=read(stream,out.data(),BLOCK);if(rc==-9981)overflows++;else check(rc);}
    ~Audio(){if(stream){stop(stream);close(stream);}if(initialized)terminate();if(dll)FreeLibrary(dll);}
};
struct Frame { int width=0,height=0;std::vector<unsigned char> pixels; };
class Camera {
    ComPtr<IMFMediaSource> source;ComPtr<IMFSourceReader> reader;LONG stride=0;int width=0,height=0;
public:
    std::wstring name;
    Camera(){
        ComPtr<IMFAttributes> attrs;hr(MFCreateAttributes(&attrs,2),"Camera attributes");
        hr(attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),"Video source type");
        IMFActivate** devices=nullptr;UINT32 count=0;hr(MFEnumDeviceSources(attrs.Get(),&devices,&count),"Enumerate cameras");
        ComPtr<IMFActivate> chosen;
        for(UINT32 i=0;i<count;i++){
            wchar_t* text=nullptr;UINT32 len=0;devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,&text,&len);
            std::wstring candidate=text?text:L"";CoTaskMemFree(text);
            std::wcout<<L"Camera: "<<candidate<<std::endl;
            if(candidate.find(L"icspring")!=std::wstring::npos){chosen=devices[i];name=candidate;}
            devices[i]->Release();
        }CoTaskMemFree(devices);
        if(!chosen)throw std::runtime_error("icspring camera not found; no automatic fallback to laptop webcam");
        hr(chosen->ActivateObject(IID_PPV_ARGS(&source)),"Activate external camera");
        attrs.Reset();hr(MFCreateAttributes(&attrs,1),"Reader attributes");
        hr(attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,TRUE),"Enable camera conversion");
        hr(MFCreateSourceReaderFromMediaSource(source.Get(),attrs.Get(),&reader),"Camera reader");
        // Choose an actual native camera mode, preferring 640x480 or 1280x720.
        ComPtr<IMFMediaType> nativeBest;double best=1e20;
        for(DWORD i=0;;i++){ComPtr<IMFMediaType> t;if(FAILED(reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,i,&t)))break;
            UINT32 w=0,h=0,n=0,d=1;MFGetAttributeSize(t.Get(),MF_MT_FRAME_SIZE,&w,&h);MFGetAttributeRatio(t.Get(),MF_MT_FRAME_RATE,&n,&d);
            double score=std::abs((double)w-640)+std::abs((double)h-480)+(d && n/d<15?10000:0);
            if(score<best){best=score;nativeBest=t;}}
        if(nativeBest)hr(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,nullptr,nativeBest.Get()),"Select camera resolution");
        ComPtr<IMFMediaType> rgb;hr(MFCreateMediaType(&rgb),"RGB type");
        rgb->SetGUID(MF_MT_MAJOR_TYPE,MFMediaType_Video);rgb->SetGUID(MF_MT_SUBTYPE,MFVideoFormat_RGB32);
        hr(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,nullptr,rgb.Get()),"Camera RGB32 conversion");
        rgb.Reset();hr(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,&rgb),"Camera current format");
        UINT32 w,h,s;hr(MFGetAttributeSize(rgb.Get(),MF_MT_FRAME_SIZE,&w,&h),"Camera dimensions");width=w;height=h;
        stride=SUCCEEDED(rgb->GetUINT32(MF_MT_DEFAULT_STRIDE,&s))?(LONG)s:width*4;
    }
    Frame capture(){
        for(int attempt=0;attempt<60;attempt++){
            ComPtr<IMFSample> sample;DWORD flags=0;LONGLONG time;
            hr(reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM,0,nullptr,&flags,&time,&sample),"Read camera frame");
            if(flags&MF_SOURCE_READERF_ENDOFSTREAM)throw std::runtime_error("Camera stream ended");
            if(!sample)continue;
            ComPtr<IMFMediaBuffer> buffer;hr(sample->ConvertToContiguousBuffer(&buffer),"Camera buffer");
            BYTE* ptr=nullptr;DWORD length=0;hr(buffer->Lock(&ptr,nullptr,&length),"Lock camera buffer");
            Frame out;out.width=width;out.height=height;out.pixels.resize((size_t)width*height*4);
            if(length<(size_t)std::abs(stride)*height){buffer->Unlock();throw std::runtime_error("Short camera frame");}
            for(int row=0;row<height;row++)memcpy(out.pixels.data()+(size_t)row*width*4,ptr+(size_t)(stride<0?height-1-row:row)*std::abs(stride),width*4);
            buffer->Unlock();return out;
        }throw std::runtime_error("Camera produced no frames");
    }
    ~Camera(){reader.Reset();if(source)source->Shutdown();}
};
struct Shared {
    std::mutex mutex;Frame frame;Estimate estimate;std::string audioError,cameraError;
    std::chrono::steady_clock::time_point audioTime{},cameraTime{};int overflows=0,frames=0;double processingMs=0;
} shared;
std::atomic<bool> running{true};double gate=-55,hfov=80;int rotation=0;bool flipX=false,flipY=false,heatmap=true;
void audioWorker(){
    CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    try{Audio audio;Beamformer beam;std::vector<float> samples;
        while(running){audio.capture(samples);auto begin=std::chrono::steady_clock::now();auto e=beam.process(samples);
            std::lock_guard<std::mutex> lock(shared.mutex);shared.estimate=std::move(e);shared.audioTime=std::chrono::steady_clock::now();shared.overflows=audio.overflows;
            shared.processingMs=std::chrono::duration<double,std::milli>(shared.audioTime-begin).count();}}
    catch(const std::exception& e){std::lock_guard<std::mutex> lock(shared.mutex);shared.audioError=e.what();}
    CoUninitialize();
}
void cameraWorker(){
    CoInitializeEx(nullptr,COINIT_MULTITHREADED);
    try{Camera camera;while(running){auto frame=camera.capture();std::lock_guard<std::mutex> lock(shared.mutex);shared.frame=std::move(frame);shared.cameraTime=std::chrono::steady_clock::now();shared.frames++;}}
    catch(const std::exception& e){std::lock_guard<std::mutex> lock(shared.mutex);shared.cameraError=e.what();}
    CoUninitialize();
}
void text(HDC dc,int x,int y,const std::wstring& s,COLORREF color=RGB(222,230,240)) {SetTextColor(dc,color);TextOutW(dc,x,y,s.c_str(),(int)s.size());}
std::wstring wide(const std::string& s){return std::wstring(s.begin(),s.end());}
void rect(HDC dc,int x,int y,int w,int h,COLORREF c){RECT r{x,y,x+w,y+h};HBRUSH b=CreateSolidBrush(c);FillRect(dc,&r,b);DeleteObject(b);}
Vec orient(double x,double y){for(int i=0;i<rotation;i++){double a=x;x=-y;y=a;}return {flipX?-x:x,flipY?-y:y,1};}
Vec unorient(double x,double y){if(flipX)x=-x;if(flipY)y=-y;for(int i=0;i<rotation;i++){double a=x;x=y;y=-a;}return{x,y,1};}
void paint(HDC target,int width,int height){
    HDC dc=CreateCompatibleDC(target);HBITMAP bitmap=CreateCompatibleBitmap(target,width,height);auto old=SelectObject(dc,bitmap);
    rect(dc,0,0,width,height,RGB(12,18,28));SetBkMode(dc,TRANSPARENT);
    HFONT font=CreateFontW(18,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Segoe UI");auto oldFont=SelectObject(dc,font);
    Frame frame;Estimate e;std::string ae,ce;int overflows,frames;double processing,age,camAge;
    {std::lock_guard<std::mutex> lock(shared.mutex);frame=shared.frame;e=shared.estimate;ae=shared.audioError;ce=shared.cameraError;overflows=shared.overflows;frames=shared.frames;processing=shared.processingMs;
        auto now=std::chrono::steady_clock::now();age=std::chrono::duration<double>(now-shared.audioTime).count();camAge=std::chrono::duration<double>(now-shared.cameraTime).count();}
    text(dc,20,15,L"UMA-16  /  LIVE SOUND DIRECTION",RGB(80,225,202));
    text(dc,20,42,L"16 microphones  |  ASIO 48 kHz  |  700 - 3200 Hz  |  Local preview");
    int vx=20,vy=82,vw=std::max(100,width-330),vh=std::max(100,height-150);
    double aspect=frame.width?(double)frame.width/frame.height:4./3;
    if((double)vw/vh>aspect)vw=(int)(vh*aspect);else vh=(int)(vw/aspect);
    rect(dc,vx,vy,vw,vh,RGB(25,35,48));
    if(!frame.pixels.empty()){
        BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=frame.width;bi.bmiHeader.biHeight=-frame.height;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;bi.bmiHeader.biCompression=BI_RGB;
        SetStretchBltMode(dc,COLORONCOLOR);StretchDIBits(dc,vx,vy,vw,vh,0,0,frame.width,frame.height,frame.pixels.data(),&bi,DIB_RGB_COLORS,SRCCOPY);
    }else {
        text(dc,vx+20,vy+30,ce.empty()?L"Waiting for external camera...":L"Camera unavailable - sound direction view remains active");
        if(!ce.empty()){
            text(dc,vx+20,vy+90,L"Close this window and double-click Start Demo.cmd from File Explorer.");
            text(dc,vx+20,vy+117,L"If access is still denied, check Settings > Privacy & security > Camera.");
            text(dc,vx+20,vy+144,L"Enable camera access for desktop apps, then start the demo again.");
        }
        text(dc,vx+20,vy+vh/2,L"LEFT");text(dc,vx+vw-65,vy+vh/2,L"RIGHT");
        text(dc,vx+vw/2-25,vy+vh-70,L"DOWN");
    }
    bool active=age<1 && e.level>=gate && e.contrast>=2.5 && !e.clipping && !e.map.empty();
    double tanH=tan(hfov*PI/360),tanV=tanH/aspect;
    if(active && heatmap){
        const int hw=160,hh=120;BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=hw;bi.bmiHeader.biHeight=-hh;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;
        void* pixels=nullptr;HBITMAP heat=CreateDIBSection(dc,&bi,DIB_RGB_COLORS,&pixels,nullptr,0);HDC hd=CreateCompatibleDC(dc);auto prior=SelectObject(hd,heat);
        double peak=*std::max_element(e.map.begin(),e.map.end()),mean=std::accumulate(e.map.begin(),e.map.end(),0.)/e.map.size();
        auto b=(BYTE*)pixels;
        for(int y=0;y<hh;y++)for(int x=0;x<hw;x++){
            auto u=unorient((2.*x/(hw-1)-1)*tanH,(1-2.*y/(hh-1))*tanV);
            int gx=(int)round((u.x/2.75+1)*.5*(GW-1)),gy=(int)round((1-u.y/2.)*.5*(GH-1));double v=0;
            if(gx>=0 && gx<GW && gy>=0 && gy<GH)v=std::clamp((e.map[gy*GW+gx]-mean)/std::max(peak-mean,1e-9),0.,1.);
            int alpha=(int)(std::pow(v,5)*130),i=(y*hw+x)*4;b[i]=0;b[i+1]=(BYTE)(alpha*.65);b[i+2]=(BYTE)alpha;b[i+3]=(BYTE)alpha;
        }
        BLENDFUNCTION bf{AC_SRC_OVER,0,255,AC_SRC_ALPHA};AlphaBlend(dc,vx,vy,vw,vh,hd,0,0,hw,hh,bf);SelectObject(hd,prior);DeleteObject(heat);DeleteDC(hd);
    }
    std::wstring status=age>=1?L"Waiting for audio":e.clipping?L"Clipping - reduce sound level":active?L"Sound direction detected":L"Quiet / diffuse sound - speak or clap";
    if(active){auto u=orient(e.x,e.y);double px=.5*(u.x/tanH+1),py=.5*(1-u.y/tanV);bool outside=px<0||px>1||py<0||py>1;
        int x=vx+(int)(std::clamp(px,.03,.97)*vw),y=vy+(int)(std::clamp(py,.04,.96)*vh);
        HPEN pen=CreatePen(PS_SOLID,3,RGB(70,255,190));auto prev=SelectObject(dc,pen);auto brush=SelectObject(dc,GetStockObject(NULL_BRUSH));
        Ellipse(dc,x-18,y-18,x+18,y+18);MoveToEx(dc,x-26,y,nullptr);LineTo(dc,x+26,y);MoveToEx(dc,x,y-26,nullptr);LineTo(dc,x,y+26);
        SelectObject(dc,brush);SelectObject(dc,prev);DeleteObject(pen);
        if(outside)status=L"Sound strongest outside camera view";
        wchar_t label[150];swprintf_s(label,L"Approx. horizontal %+.0f deg  /  vertical %+.0f deg",atan(u.x)*180/PI,atan2(u.y,sqrt(1+u.x*u.x))*180/PI);
        rect(dc,vx,vy+vh-28,vw,28,RGB(12,18,28));text(dc,vx+8,vy+vh-25,label);
    }
    int sx=width-290;text(dc,sx,84,L"MICROPHONE LEVELS",RGB(80,225,202));
    for(int i=0;i<CH;i++){int y=116+i*20;wchar_t label[64];swprintf_s(label,L"%02d",i+1);text(dc,sx,y,label);rect(dc,sx+32,y+4,145,11,RGB(35,47,62));
        double db=age<1?e.db[i]:-100;int n=(int)(std::clamp((db+85)/65,0.,1.)*145);rect(dc,sx+32,y+4,n,11,db>-3?RGB(255,80,80):RGB(64,210,170));swprintf_s(label,L"%5.1f",db);text(dc,sx+185,y,label);}
    wchar_t info[160];swprintf_s(info,L"Gate: %.0f dBFS  |  FOV: %.0f deg",gate,hfov);text(dc,sx,455,info);
    swprintf_s(info,L"Capture overruns: %d",overflows);text(dc,sx,480,info);
    swprintf_s(info,L"Processing: %.0f ms  |  Video: %d frames",processing,frames);text(dc,sx,505,info);
    text(dc,sx,540,L"[ / ]  sensitivity gate");text(dc,sx,563,L"- / +  camera field of view");
    text(dc,sx,586,L"R rotate   X / Y flip direction");text(dc,sx,609,L"H heatmap   Esc quit");
    text(dc,20,height-56,status,RGB(80,225,202));
    text(dc,20,height-30,L"Approximate direction only. Camera alignment/FOV uncalibrated; front/back ambiguity.",RGB(170,180,195));
    if(camAge>2 && !frame.pixels.empty())text(dc,vx+12,vy+12,L"CAMERA STALE",RGB(255,110,90));
    if(!ae.empty())text(dc,20,70,wide("Audio error: "+ae),RGB(255,110,90));
    if(!ce.empty())text(dc,vx+12,vy+60,wide("Camera error: "+ce),RGB(255,110,90));
    BitBlt(target,0,0,width,height,dc,0,0,SRCCOPY);SelectObject(dc,oldFont);DeleteObject(font);SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
}
LRESULT CALLBACK windowProc(HWND w,UINT m,WPARAM k,LPARAM l){
    if(m==WM_PAINT){PAINTSTRUCT p;HDC dc=BeginPaint(w,&p);RECT r;GetClientRect(w,&r);paint(dc,r.right,r.bottom);EndPaint(w,&p);return 0;}
    if(m==WM_TIMER){
        static int ticks=0;
        if(++ticks%30==0){std::lock_guard<std::mutex> lock(shared.mutex);std::ofstream out("live-status.txt");
            out<<"Audio age (seconds): "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-shared.audioTime).count()<<"\n";
            out<<"Level dBFS: "<<shared.estimate.level<<"\nContrast: "<<shared.estimate.contrast<<"\nProcessing ms: "<<shared.processingMs<<"\n";
            out<<"Audio overruns: "<<shared.overflows<<"\nCamera frames: "<<shared.frames<<"\nAudio error: "<<shared.audioError<<"\nCamera error: "<<shared.cameraError<<"\n";}
        InvalidateRect(w,nullptr,FALSE);return 0;}
    if(m==WM_GETMINMAXINFO){auto info=(MINMAXINFO*)l;info->ptMinTrackSize={1000,740};return 0;}
    if(m==WM_KEYDOWN){if(k==VK_ESCAPE)DestroyWindow(w);if(k=='X')flipX=!flipX;if(k=='Y')flipY=!flipY;if(k=='R')rotation=(rotation+1)%4;if(k=='H')heatmap=!heatmap;
        if(k==VK_OEM_4)gate=std::max(-90.,gate-3);if(k==VK_OEM_6)gate=std::min(-20.,gate+3);if(k==VK_OEM_MINUS)hfov=std::max(30.,hfov-5);if(k==VK_OEM_PLUS)hfov=std::min(140.,hfov+5);return 0;}
    if(m==WM_DESTROY){running=false;PostQuitMessage(0);return 0;}return DefWindowProcW(w,m,k,l);
}
void selfTest(){
    Beamformer beam;auto pos=positions();
    for(Vec source:std::vector<Vec>{{.66,.35,1},{-.88,-.47,1},{0,0,1}}){
        double norm=sqrt(source.x*source.x+source.y*source.y+1);Vec u{source.x/norm,source.y/norm,1/norm};std::vector<float> audio(BLOCK*CH);
        for(int t=0;t<BLOCK;t++)for(int c=0;c<CH;c++){
            double delay=(pos[c].x*u.x+pos[c].y*u.y)/SPEED,v=0;
            for(int f=0;f<31;f++){double freq=750+f*79.13;v+=sin(2*PI*freq*(t/FS+delay)+f*f*.713);}
            audio[t*CH+c]=(float)(v*.002);}
        auto e=beam.process(audio);std::cout<<"Synthetic source "<<source.x<<","<<source.y<<" -> "<<e.x<<","<<e.y<<" contrast="<<e.contrast<<std::endl;
        if(std::abs(e.x-source.x)>.17 || std::abs(e.y-source.y)>.17)throw std::runtime_error("Beamforming direction self-test failed");
    }
    std::vector<float> silence(BLOCK*CH,0);auto e=beam.process(silence);if(e.level>-100 || e.contrast>1)throw std::runtime_error("Silence gate test failed");
    std::cout<<"PASS: left/right, vertical, center, silence tests"<<std::endl;
}
int main(int argc,char** argv){
    try{
        if(argc>1 && std::string(argv[1])=="--self-test"){selfTest();return 0;}
        hr(CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED),"COM initialization");hr(MFStartup(MF_VERSION),"Media Foundation initialization");
        if(argc>1 && std::string(argv[1])=="--check"){
            Frame f;bool cameraOk=false;
            try {Camera camera;f=camera.capture();cameraOk=true;std::cout<<"Camera frame: "<<f.width<<" x "<<f.height<<", "<<f.pixels.size()<<" bytes"<<std::endl;}
            catch(const std::exception& error){std::cout<<"Camera check failed: "<<error.what()<<std::endl;shared.cameraError=error.what();}
            Audio audio;Beamformer beam;std::vector<float> samples;Estimate e;double processing=0;
            for(int i=0;i<24;i++){audio.capture(samples);auto begin=std::chrono::steady_clock::now();e=beam.process(samples);processing+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
            std::cout<<"Mean processing per 171 ms audio block: "<<processing/24<<" ms"<<std::endl;
            std::cout<<"Audio: "<<audio.name<<", 16 channels, 48000 Hz, overruns="<<audio.overflows<<std::endl;
            for(int c=0;c<CH;c++)std::cout<<"Mic "<<c+1<<": "<<e.db[c]<<" dBFS"<<std::endl;
            if(argc>2 && std::string(argv[2])=="--save-preview"){
                {std::lock_guard<std::mutex> lock(shared.mutex);shared.frame=f;shared.estimate=e;shared.audioTime=shared.cameraTime=std::chrono::steady_clock::now();shared.frames=cameraOk?1:0;shared.processingMs=processing/24;}
                const int w=1250,h=760;BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=w;bi.bmiHeader.biHeight=-h;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;
                void* data;HDC dc=CreateCompatibleDC(nullptr);HBITMAP bitmap=CreateDIBSection(dc,&bi,DIB_RGB_COLORS,&data,nullptr,0);auto old=SelectObject(dc,bitmap);paint(dc,w,h);
                BITMAPFILEHEADER fh{};fh.bfType=0x4D42;fh.bfOffBits=sizeof(fh)+sizeof(BITMAPINFOHEADER);fh.bfSize=fh.bfOffBits+w*h*4;
                std::ofstream file("preview.bmp",std::ios::binary);file.write((char*)&fh,sizeof(fh));file.write((char*)&bi.bmiHeader,sizeof(BITMAPINFOHEADER));file.write((char*)data,w*h*4);
                SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
            }
            return cameraOk?0:2;
        }
        SetProcessDPIAware();HINSTANCE instance=GetModuleHandle(nullptr);WNDCLASSW wc{};wc.lpfnWndProc=windowProc;wc.hInstance=instance;wc.lpszClassName=L"UMAMicDemo";wc.hCursor=LoadCursor(nullptr,IDC_ARROW);RegisterClassW(&wc);
        HWND window=CreateWindowW(wc.lpszClassName,L"UMA-16 + Camera | Live Sound Direction",WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,1280,820,nullptr,nullptr,instance,nullptr);
        if(!window)throw std::runtime_error("Could not create demo window");
        std::thread at(audioWorker),ct(cameraWorker);ShowWindow(window,SW_SHOW);SetTimer(window,1,66,nullptr);
        MSG msg;while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}
        running=false;at.join();ct.join();MFShutdown();CoUninitialize();return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
