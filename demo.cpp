#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#include <setupapi.h>
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
std::string utf8(const std::wstring& s);
std::string arrayHardwareId(){
    const GUID cls={0xa4b5a684,0x87e4,0x44bf,{0x88,0x4f,0x84,0x31,0x78,0x4e,0x4b,0xd6}};
    HDEVINFO devices=SetupDiGetClassDevsW(&cls,nullptr,nullptr,DIGCF_PRESENT);
    if(devices==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot identify the connected miniDSP array");
    std::string selected;int matches=0;
    for(DWORD i=0;;i++){SP_DEVINFO_DATA info{};info.cbSize=sizeof(info);if(!SetupDiEnumDeviceInfo(devices,i,&info))break;
        wchar_t id[1024]{};if(SetupDiGetDeviceInstanceIdW(devices,&info,id,1024,nullptr)){
            std::wstring value=id;if(value.find(L"VID_2752&PID_0019")!=std::wstring::npos){selected=utf8(value);matches++;}}}
    SetupDiDestroyDeviceInfoList(devices);
    if(matches!=1)throw std::runtime_error("Connect exactly one UMA-16 v2 array before starting setup");return selected;
}
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
#include "tone_locator.h"
struct Estimate { std::array<double,CH> db{}; std::vector<double> map; double x=0,y=0,level=-120,contrast=0; bool clipping=false; ToneEstimate tone; };
struct Pair { int a,b; std::vector<int> lag; std::vector<double> frac; };
class Beamformer {
    std::array<Vec,CH> pos=positions(); std::vector<Pair> pairs;
    ToneLocator toneLocator;
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
        Estimate out;out.tone=toneLocator.process(audio);out.map.assign(GW*GH,0);double energy=0;
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
    std::string name,hardwareId;int overflows=0;
    Audio(){try {
        hardwareId=arrayHardwareId();
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
    } catch(...) {if(stream&&close)close(stream);if(initialized&&terminate)terminate();if(dll)FreeLibrary(dll);throw;}}
    void capture(std::vector<float>& out){out.resize(BLOCK*CH);int rc=read(stream,out.data(),BLOCK);if(rc==-9981)overflows++;else check(rc);}
    ~Audio(){if(stream){stop(stream);close(stream);}if(initialized)terminate();if(dll)FreeLibrary(dll);}
};
struct Frame { int width=0,height=0;std::vector<unsigned char> pixels; };
class Camera {
    ComPtr<IMFMediaSource> source;ComPtr<IMFSourceReader> reader;LONG stride=0;int width=0,height=0;
public:
    std::wstring name;std::string hardwareId;
    Camera(){try {
        ComPtr<IMFAttributes> attrs;hr(MFCreateAttributes(&attrs,2),"Camera attributes");
        hr(attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),"Video source type");
        IMFActivate** devices=nullptr;UINT32 count=0;hr(MFEnumDeviceSources(attrs.Get(),&devices,&count),"Enumerate cameras");
        ComPtr<IMFActivate> chosen;
        for(UINT32 i=0;i<count;i++){
            wchar_t* text=nullptr;UINT32 len=0;devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,&text,&len);
            std::wstring candidate=text?text:L"";CoTaskMemFree(text);
            std::wcout<<L"Camera: "<<candidate<<std::endl;
            if(candidate.find(L"icspring")!=std::wstring::npos){chosen=devices[i];name=candidate;
                wchar_t* link=nullptr;UINT32 length=0;
                if(SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,&link,&length))){hardwareId=utf8(link);CoTaskMemFree(link);}}
            devices[i]->Release();
        }CoTaskMemFree(devices);
        if(!chosen||hardwareId.empty())throw std::runtime_error("icspring camera not found; no automatic fallback to laptop webcam");
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
    }catch(...){reader.Reset();if(source)source->Shutdown();throw;}}
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
#include "wizard_app.h"
#include "wizard_tests.h"
#include "tone_tests.h"
#include "focus_app.h"
#include "focus_tests.h"

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
        wchar_t executable[MAX_PATH];GetModuleFileNameW(nullptr,executable,MAX_PATH);std::wstring folder(executable);folder=folder.substr(0,folder.find_last_of(L"\\/"));SetCurrentDirectoryW(folder.c_str());
        if(argc>1 && std::string(argv[1])=="--self-test"){selfTest();return 0;}
        if(argc>1 && std::string(argv[1])=="--setup-tests"){setupTests();return 0;}
        if(argc>1 && std::string(argv[1])=="--tone-tests"){toneTests();return 0;}
        if(argc>1 && std::string(argv[1])=="--make-phone-tone"){writePhoneTone();return 0;}
        if(argc>1 && std::string(argv[1])=="--focus-tests"){CreateDirectoryW(L"test-output",nullptr);focusTests();return 0;}
        if(argc>1 && std::string(argv[1])=="--render-focus-tests"){renderFocusTests();return 0;}
        if(argc>1 && std::string(argv[1])=="--focus-capture-check"){return focusCaptureCheck();}
        if(argc>1 && std::string(argv[1])=="--render-wizard-tests"){renderWizardTests();return 0;}
        if(argc>1 && std::string(argv[1])=="--array-id"){std::cout<<arrayHardwareId()<<std::endl;return 0;}
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
        HANDLE singleton=CreateMutexW(nullptr,FALSE,L"Local\\IkaMicArrayWizardV1");DWORD mutexError=GetLastError();
        if(!singleton||mutexError==ERROR_ALREADY_EXISTS){HWND existing=FindWindowW(L"IkaMicWizard",nullptr);if(existing){ShowWindow(existing,SW_RESTORE);SetForegroundWindow(existing);}else MessageBoxW(nullptr,L"The setup app is already running. Close the existing instance before starting another.",L"Ika microphone setup",MB_OK|MB_ICONINFORMATION);if(singleton)CloseHandle(singleton);return 0;}
        if(FindWindowW(L"UMAMicDemo",nullptr)){MessageBoxW(nullptr,L"Close the basic microphone demo first, then launch the setup wizard. Both apps use the same ASIO microphone.",L"Close the basic demo",MB_OK|MB_ICONINFORMATION);CloseHandle(singleton);return 2;}
        if(argc==1||std::string(argv[1])=="--focus"){int result=focus::run();CloseHandle(singleton);MFShutdown();CoUninitialize();return result;}
        wizard.load(argc>1&&std::string(argv[1])=="--wizard");
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);HINSTANCE instance=GetModuleHandle(nullptr);WNDCLASSW wc{};wc.lpfnWndProc=windowProc;wc.hInstance=instance;wc.lpszClassName=L"IkaMicWizard";wc.hCursor=LoadCursor(nullptr,IDC_ARROW);RegisterClassW(&wc);
        RECT work{};SystemParametersInfoW(SPI_GETWORKAREA,0,&work,0);double dpiScale=GetDpiForSystem()/96.;
        int windowWidth=std::min((int)(1360*dpiScale),(int)(work.right-work.left)),windowHeight=std::min((int)(940*dpiScale),(int)(work.bottom-work.top));
        HWND window=CreateWindowW(wc.lpszClassName,L"Ika | Microphone + Camera Setup",WS_OVERLAPPEDWINDOW,work.left+(work.right-work.left-windowWidth)/2,work.top+(work.bottom-work.top-windowHeight)/2,windowWidth,windowHeight,nullptr,nullptr,instance,nullptr);
        if(!window)throw std::runtime_error("Could not create demo window");
        std::thread at(audioWorker),ct(cameraWorker);ShowWindow(window,SW_SHOW);SetTimer(window,1,66,nullptr);
        MSG msg;while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}
        running=false;at.join();ct.join();CloseHandle(singleton);MFShutdown();CoUninitialize();return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
