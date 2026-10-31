#pragma once
#include "setup_model.h"
#include <windowsx.h>

using Clock=std::chrono::steady_clock;
double secondsNow(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
std::wstring wide(const std::string& s){if(s.empty())return {};int n=MultiByteToWideChar(CP_UTF8,0,s.data(),(int)s.size(),nullptr,0);std::wstring out(n,0);MultiByteToWideChar(CP_UTF8,0,s.data(),(int)s.size(),out.data(),n);return out;}
std::string utf8(const std::wstring& s){if(s.empty())return {};int n=WideCharToMultiByte(CP_UTF8,0,s.data(),(int)s.size(),nullptr,0,nullptr,nullptr);std::string out(n,0);WideCharToMultiByte(CP_UTF8,0,s.data(),(int)s.size(),out.data(),n,nullptr,nullptr);return out;}
std::string jsonString(const std::string& s){std::string out="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=(char)c;}else if(c<32){char b[8];sprintf_s(b,"\\u%04x",c);out+=b;}else out+=(char)c;}return out+'"';}
struct Shared {
    std::mutex mutex;Frame frame;Estimate estimate;std::string audioError,cameraError,audioId,cameraId;
    Clock::time_point audioTime{},cameraTime{};int overflows=0,frames=0;double processingMs=0;unsigned long long sequence=0;
} shared;
struct Snapshot {
    Estimate estimate;std::string audioError,cameraError,audioId,cameraId;int width=0,height=0,frames=0,overflows=0;
    double audioAge=1e9,cameraAge=1e9,processingMs=0;unsigned long long sequence=0;
    bool hardwareReady()const {
        if(audioAge>.75||cameraAge>2||width<160||frames<5||sequence<10||audioId.empty()||cameraId.empty()||!audioError.empty()||!cameraError.empty()||estimate.clipping||!std::isfinite(estimate.level))return false;
        for(double db:estimate.db)if(!std::isfinite(db)||db< -105)return false;return true;
    }
};
Snapshot snapshot(){
    Snapshot s;std::lock_guard<std::mutex> lock(shared.mutex);s.estimate=shared.estimate;s.audioError=shared.audioError;s.cameraError=shared.cameraError;s.audioId=shared.audioId;s.cameraId=shared.cameraId;
    s.width=shared.frame.width;s.height=shared.frame.height;s.frames=shared.frames;s.overflows=shared.overflows;s.processingMs=shared.processingMs;s.sequence=shared.sequence;
    auto now=Clock::now();if(shared.audioTime.time_since_epoch().count())s.audioAge=std::chrono::duration<double>(now-shared.audioTime).count();if(shared.cameraTime.time_since_epoch().count())s.cameraAge=std::chrono::duration<double>(now-shared.cameraTime).count();return s;
}
std::atomic<bool> running{true};std::atomic<unsigned> retryDevices{0};
void waitForRetry(unsigned generation){while(running && retryDevices==generation)std::this_thread::sleep_for(std::chrono::milliseconds(100));}
void audioWorker(){
    CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    while(running){unsigned generation=retryDevices;
        try{Audio audio;Beamformer beam;std::vector<float> samples;
            {std::lock_guard<std::mutex> lock(shared.mutex);shared.audioId=audio.hardwareId;shared.audioError.clear();}
            while(running){audio.capture(samples);auto begin=Clock::now();auto e=beam.process(samples);
                std::lock_guard<std::mutex> lock(shared.mutex);shared.estimate=std::move(e);shared.audioTime=Clock::now();shared.overflows=audio.overflows;shared.sequence++;
                shared.processingMs=std::chrono::duration<double,std::milli>(shared.audioTime-begin).count();}}
        catch(const std::exception& e){std::lock_guard<std::mutex> lock(shared.mutex);shared.audioError=e.what();}
        if(running)waitForRetry(generation);
    }CoUninitialize();
}
void cameraWorker(){
    CoInitializeEx(nullptr,COINIT_MULTITHREADED);
    while(running){unsigned generation=retryDevices;
        try{Camera camera;
            {std::lock_guard<std::mutex> lock(shared.mutex);shared.cameraId=camera.hardwareId;shared.cameraError.clear();}
            while(running){auto frame=camera.capture();std::lock_guard<std::mutex> lock(shared.mutex);shared.frame=std::move(frame);shared.cameraTime=Clock::now();shared.frames++;}}
        catch(const std::exception& e){std::lock_guard<std::mutex> lock(shared.mutex);shared.cameraError=e.what();}
        if(running)waitForRetry(generation);
    }CoUninitialize();
}
enum class Step {Devices,Noise,Align,Verify,Zone,Trial,Review,Live};
enum class Capture {None,Noise,Align,Verify,Inside,Outside};
struct Button {RECT bounds;int action;bool enabled;};
enum Action {Primary=1,Back,Retry,Restart,Inside,Outside,MoreSensitive,LessSensitive,SaveSensitivity,ToggleHeatmap,ToggleSource};
struct Wizard {
    Step step=Step::Devices;Capture capture=Capture::None;setup::Profile profile;setup::NoiseResult room;
    std::vector<setup::Observation> points;std::vector<setup::Point> samples;
    std::vector<std::array<double,16>> noiseSamples;std::vector<double> noiseLevels;
    setup::Point clicked,dragStart;bool targetSet=false,dragging=false,verified=false,insideDone=false,outsideDone=false,loaded=false,loadPending=false,dirty=false;
    bool clipped=false,heatmap=true,phoneTone=true;int total=0,inRegion=0;double began=0,validation=1;unsigned long long lastSequence=0;
    setup::Tracker tracker;std::string note="Close the basic demo first so this app can use the array and camera.";
    std::vector<Button> buttons;RECT video{};
    void reset(){*this=Wizard{};}
    bool collecting() const{return capture!=Capture::None;}
    double duration() const {return capture==Capture::Noise?6.:capture==Capture::Inside||capture==Capture::Outside?8.:4.;}
    double elapsed() const{return secondsNow()-began-3.;}
    bool profileMatches(const Snapshot& s)const{return profile.matches(s.audioId,s.cameraId,s.width,s.height);}
    bool usable(const Snapshot& s)const{return s.audioAge<.75&&s.audioError.empty()&&!s.estimate.clipping&&std::isfinite(s.estimate.level)&&std::isfinite(s.estimate.contrast)&&std::isfinite(s.estimate.x)&&std::isfinite(s.estimate.y)&&s.estimate.level>=profile.gate&&s.estimate.contrast>=2.0;}
    bool toneAlignment()const{return phoneTone&&(step==Step::Align||step==Step::Verify);}
    bool alignmentUsable(const Snapshot& s)const{return phoneTone?s.audioAge<.75&&s.audioError.empty()&&!s.estimate.clipping&&s.estimate.tone.valid&&std::isfinite(s.estimate.tone.x)&&std::isfinite(s.estimate.tone.y):usable(s);}
    setup::Point alignmentPoint(const Snapshot& s)const{return phoneTone?setup::Point{s.estimate.tone.x,s.estimate.tone.y}:setup::Point{s.estimate.x,s.estimate.y};}
    bool nextEnabled(const Snapshot& s)const {
        if(collecting())return false;
        switch(step){case Step::Devices:return s.hardwareReady();case Step::Noise:return s.hardwareReady();
            case Step::Align:return s.hardwareReady()&&(points.size()>=4||targetSet);case Step::Verify:return s.hardwareReady()&&(verified||targetSet);
            case Step::Zone:return s.hardwareReady()&&profile.region.valid();case Step::Trial:return s.hardwareReady()&&insideDone&&outsideDone;
            case Step::Review:return s.hardwareReady()&&profileMatches(s)&&profile.valid();case Step::Live:return true;}return false;
    }
    void load(bool forceWizard){std::ifstream file("setup-profile.ini");if(!file)return;
        if(!setup::deserialize(file,profile)){note="The saved profile is incomplete or invalid. Complete setup to replace it.";return;}
        loaded=true;loadPending=!forceWizard;note=forceWizard?"Existing profile kept until you save a replacement.":"Checking the saved setup against the connected hardware...";
    }
    bool save(){
        if(!profile.valid()){note="Setup is incomplete. Finish the verification and interaction tests.";return false;}
        {std::ofstream out("setup-profile.ini.tmp",std::ios::trunc);setup::serialize(out,profile);out.flush();if(!out){note="Could not write setup-profile.ini.tmp. Check folder permissions.";return false;}}
        if(!MoveFileExW(L"setup-profile.ini.tmp",L"setup-profile.ini",MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){note="Could not save the profile. The previous profile was kept.";return false;}
        dirty=false;loaded=true;note="Setup saved for this array and camera. Speech classification is not enabled yet.";return true;
    }
    void start(Capture kind,const Snapshot& s){
        if(!s.hardwareReady()){note="Wait for fresh camera video and all 16 microphones before capturing.";return;}
        if(!profileMatches(s)){note="Connected devices or camera resolution changed. Restart setup.";return;}
        capture=kind;began=secondsNow();lastSequence=s.sequence;total=inRegion=0;clipped=false;samples.clear();noiseSamples.clear();noiseLevels.clear();
        note="Get into position. Capture begins after the three-second countdown.";
    }
    void finish(){
        auto kind=capture;capture=Capture::None;
        if(kind==Capture::Noise){room=setup::measureNoise(noiseSamples,noiseLevels,clipped);note=room.message;
            if(room.good){profile.noise=room.floor;profile.gate=room.gate;profile.channels=room.channels;points.clear();verified=insideDone=outsideDone=false;profile.validation=1;}}
        else if(kind==Capture::Align||kind==Capture::Verify){
            auto r=setup::summarize(samples,total);note=r.message;if(!r.good){if(phoneTone&&(samples.size()<10||total<10||(double)samples.size()/std::max(1,total)<.4))note="Too little clear 440 Hz tone. Play a continuous sine tone, aim the phone speaker at the array and increase its volume slightly. Wait for tone lock, then retry.";return;}
            if(kind==Capture::Align){points.push_back({r.median,clicked,r.spread});targetSet=false;
                if(points.size()==4){std::string error;if(!setup::fit(points,profile.projection,error)){note=error;points.clear();}else note="Four points captured. Continue to verify at a new position.";}
                else note="Position captured. Move the source to the next suggested area, then click it.";
            }else {auto p=profile.projection.project(r.median);validation=hypot(p.x-clicked.x,p.y-clicked.y);verified=validation<=.10;
                if(verified){profile.validation=validation;note="Independent position passed. Continue to select the interaction area.";}
                else note="Marker missed the new position. Retry, or go back to redo alignment. The profile is not verified.";}
        }else {
            if(total<20||samples.size()<10||(double)samples.size()/total<.25){note="Too little clear sound for this test. Speak continuously or rustle paper, then retry.";return;}
            double ratio=(double)inRegion/samples.size();
            if(kind==Capture::Inside){profile.insideRate=ratio;insideDone=ratio>=.60;note=insideDone?"Inside test passed. Now make a similar sound outside the outlined area.":"Too many directions missed the area. Retry, widen the area, or redo alignment.";}
            else {profile.outsideRate=ratio;outsideDone=ratio<=.35;note=outsideDone?"Outside test passed. You can review and save setup.":"Outside sound often landed inside. Retry farther from the boundary or redo alignment.";}
        }
    }
    void tick(const Snapshot& s){
        if(loadPending&&s.hardwareReady()){
            loadPending=false;if(profileMatches(s)){step=Step::Live;note="Saved setup loaded. This detects sound location, not confirmed speech.";}
            else {note="Hardware or resolution differs from the saved profile. Please complete setup again.";loaded=false;}
        }
        if(collecting()&&(!s.hardwareReady()||!profileMatches(s))){capture=Capture::None;note="Capture stopped: hardware changed or input became stale. Check devices, then retry.";}
        if(s.sequence!=lastSequence){lastSequence=s.sequence;
            if(collecting()&&elapsed()>=0&&elapsed()<=duration()){
                total++;clipped|=s.estimate.clipping;
                if(capture==Capture::Noise){noiseSamples.push_back(s.estimate.db);noiseLevels.push_back(s.estimate.level);}
                else if(capture==Capture::Align||capture==Capture::Verify){if(alignmentUsable(s))samples.push_back(alignmentPoint(s));}
                else if(usable(s)){setup::Point p{s.estimate.x,s.estimate.y};samples.push_back(p);if(profile.region.contains(profile.projection.project(p)))inRegion++;}
            }
            if(step==Step::Live)tracker.update(s.hardwareReady()&&profileMatches(s)&&usable(s),profile.projection.project({s.estimate.x,s.estimate.y}),secondsNow());
        }
        if(step==Step::Live&&(!s.hardwareReady()||!profileMatches(s)))tracker.reset();
        if(collecting()&&elapsed()>duration())finish();
    }
    void primary(const Snapshot& s){
        if(!nextEnabled(s))return;
        if(step==Step::Devices){loadPending=false;profile=setup::Profile{};profile.audioId=s.audioId;profile.cameraId=s.cameraId;profile.width=s.width;profile.height=s.height;step=Step::Noise;note="Keep fans and normal room sounds as they will be during use. Stay quiet while measuring.";}
        else if(step==Step::Noise){if(room.good){step=Step::Align;targetSet=false;note=phoneTone?"Play a continuous 440 Hz sine tone on your phone. Hold it at least 1 metre away, speaker facing the array. Wait for tone lock, then click the phone speaker in the video.":"Use one steady sound at least 1 metre away. Click its actual position in the video.";}else start(Capture::Noise,s);}
        else if(step==Step::Align){if(points.size()==4){step=Step::Verify;targetSet=false;note="Choose a NEW sound position near the center. This point was not used to fit the alignment.";}else start(Capture::Align,s);}
        else if(step==Step::Verify){if(verified){step=Step::Zone;note="Stop the phone tone. Drag a rectangle over the area where a person should interact with Ika.";}else start(Capture::Verify,s);}
        else if(step==Step::Zone){step=Step::Trial;insideDone=outsideDone=false;note="Run both tests with one person/source in front of the array. These test location, not whether a sound is speech.";}
        else if(step==Step::Trial){step=Step::Review;note="Review the measurements. Save enables the live calibrated view and local status output.";}
        else if(step==Step::Review){if(save()){step=Step::Live;tracker.reset();}}
    }
    void action(int id,const Snapshot& s){
        if(id==ToggleSource){if(collecting()||(step!=Step::Align&&step!=Step::Verify))return;phoneTone=!phoneTone;points.clear();samples.clear();verified=insideDone=outsideDone=targetSet=false;profile.validation=1;step=Step::Align;note=phoneTone?"440 Hz phone mode selected. Play a continuous sine tone and wait for tone lock. Previous alignment cleared.":"Broadband mode selected. Use steady speech or rustling paper. Previous alignment cleared.";return;}
        if(id==ToggleHeatmap){heatmap=!heatmap;return;}
        if(id==Primary){primary(s);return;}
        if(id==Retry){retryDevices++;note="Retrying failed devices. Close other apps using this array or camera.";return;}
        if(id==Restart){reset();note="Starting a new setup. Your saved profile stays intact until Save.";return;}
        if(id==Back){capture=Capture::None;targetSet=false;
            if(step==Step::Noise){step=Step::Devices;room={};}
            else if(step==Step::Align){step=Step::Noise;room={};points.clear();}
            else if(step==Step::Verify){step=Step::Align;points.clear();verified=false;}
            else if(step==Step::Zone){step=Step::Verify;verified=false;}
            else if(step==Step::Trial){step=Step::Zone;insideDone=outsideDone=false;}
            else if(step==Step::Review)step=Step::Trial;
            note="Previous step opened. Repeat the measurement before continuing.";return;
        }
        if(id==Inside||id==Outside){if(collecting())return;if(id==Inside)insideDone=false;else outsideDone=false;start(id==Inside?Capture::Inside:Capture::Outside,s);return;}
        if(id==MoreSensitive||id==LessSensitive){profile.gate=std::clamp(profile.gate+(id==MoreSensitive?-2.:2.),-90.,-15.);dirty=true;note="Sensitivity changed. Save it to keep it for the next run.";return;}
        if(id==SaveSensitivity){step=Step::Trial;insideDone=outsideDone=false;note="Sensitivity changed. Repeat inside and outside tests before saving the new gate.";}
    }
} wizard;

void text(HDC dc,int x,int y,const std::wstring& s,COLORREF color=RGB(222,230,240)){SetTextColor(dc,color);TextOutW(dc,x,y,s.c_str(),(int)s.size());}
void wrapped(HDC dc,RECT r,const std::wstring& s,COLORREF color=RGB(190,205,222)){SetTextColor(dc,color);DrawTextW(dc,s.c_str(),(int)s.size(),&r,DT_LEFT|DT_WORDBREAK|DT_NOPREFIX);}
void rect(HDC dc,int x,int y,int w,int h,COLORREF c){RECT r{x,y,x+w,y+h};HBRUSH b=CreateSolidBrush(c);FillRect(dc,&r,b);DeleteObject(b);}
void outline(HDC dc,RECT r,COLORREF c,int width=2){HPEN p=CreatePen(PS_SOLID,width,c);auto old=SelectObject(dc,p);auto brush=SelectObject(dc,GetStockObject(NULL_BRUSH));Rectangle(dc,r.left,r.top,r.right,r.bottom);SelectObject(dc,brush);SelectObject(dc,old);DeleteObject(p);}
std::wstring number(double n,int precision=1){std::wostringstream out;out<<std::fixed<<std::setprecision(precision)<<n;return out.str();}
void button(HDC dc,RECT r,const std::wstring& label,int action,bool enabled=true,bool accent=false){
    wizard.buttons.push_back({r,action,enabled});rect(dc,r.left,r.top,r.right-r.left,r.bottom-r.top,enabled?(accent?RGB(25,119,109):RGB(39,55,73)):RGB(28,35,44));
    SetTextColor(dc,enabled?RGB(235,245,250):RGB(100,115,130));DrawTextW(dc,label.c_str(),(int)label.size(),&r,DT_CENTER|DT_VCENTER|DT_SINGLELINE);
}
RECT pixelRect(setup::Region region){auto r=wizard.video;int w=r.right-r.left,h=r.bottom-r.top;return {r.left+(int)(region.left*w),r.top+(int)(region.top*h),r.left+(int)(region.right*w),r.top+(int)(region.bottom*h)};}
void crosshair(HDC dc,setup::Point point,COLORREF color,bool dashed=false){
    auto r=wizard.video;int x=r.left+(int)(std::clamp(point.x,0.,1.)*(r.right-r.left)),y=r.top+(int)(std::clamp(point.y,0.,1.)*(r.bottom-r.top));
    HPEN p=CreatePen(dashed?PS_DOT:PS_SOLID,1,color);auto old=SelectObject(dc,p);auto brush=SelectObject(dc,GetStockObject(NULL_BRUSH));
    Ellipse(dc,x-20,y-20,x+20,y+20);MoveToEx(dc,x-28,y,nullptr);LineTo(dc,x+28,y);MoveToEx(dc,x,y-28,nullptr);LineTo(dc,x,y+28);
    SelectObject(dc,brush);SelectObject(dc,old);DeleteObject(p);
}
std::wstring primaryLabel(){switch(wizard.step){
    case Step::Devices:return L"Begin setup";case Step::Noise:return wizard.room.good?L"Continue to alignment":L"Measure room (6 sec)";
    case Step::Align:return wizard.points.size()==4?L"Verify alignment":L"Capture position (4 sec)";
    case Step::Verify:return wizard.verified?L"Choose interaction area":L"Check new position (4 sec)";
    case Step::Zone:return L"Test interaction area";case Step::Trial:return L"Review setup";case Step::Review:return L"Save & start live view";default:return L"";}}
std::wstring instruction(){switch(wizard.step){
    case Step::Devices:return L"Check that the external camera is live and all 16 microphone bars respond. Close the basic demo before continuing.";
    case Step::Noise:return L"Stop the phone tone. Keep normal room noise running. Stay quiet yourself for six seconds to measure the background.";
    case Step::Align:return wizard.phoneTone?L"Play a continuous 440 Hz sine tone on your phone, at least 1 metre away. Move LEFT, RIGHT, HIGH, then LOW. Wait for tone lock; click the phone speaker, then capture.":L"Place one steady sound LEFT, RIGHT, HIGH, then LOW in the image, at least 1 metre away. Click its actual location, then capture.";
    case Step::Verify:return wizard.phoneTone?L"Keep the 440 Hz tone playing. Move the phone to a NEW position near the center. Wait for tone lock, click its speaker, then capture to check the alignment.":L"Move the sound to a NEW position near the center. Click the visible source, then capture. We will check the alignment on this unused point.";
    case Step::Zone:return L"Drag a rectangle in the video around Ika's interaction area. It should contain where a person's face/upper body will be when speaking.";
    case Step::Trial:return L"Stop the phone tone. First speak steadily inside the area, then outside it, still in front of the array. These tests use normal sound and check location only.";
    case Step::Review:return L"The setup is ready to save. Alignment is empirical, not a full lens calibration. A sound in the area is not yet proof that a person is speaking.";
    case Step::Live:return L"Speak or rustle paper in front of the array. Green marks sound in the interaction area; orange marks sound outside. Speech classification is not enabled.";
    }return L"";
}
// Keep the entire interface readable as the preview grows. Store geometry in
// logical units and use the same transform for rendering and mouse input.
struct UiLayout {
    int width=1280,height=840;
    double scaleX=1,scaleY=1;
    void resize(int pixelsWide,int pixelsHigh){
        double scale=std::min(pixelsWide/1280.,pixelsHigh/840.);
        width=(int)std::ceil(pixelsWide/scale);height=(int)std::ceil(pixelsHigh/scale);
        scaleX=(double)pixelsWide/width;scaleY=(double)pixelsHigh/height;
    }
    POINT logical(int x,int y)const{return {(LONG)std::floor(x/scaleX),(LONG)std::floor(y/scaleY)};}
} uiLayout;
POINT clientPoint(HWND window,LPARAM position){
    RECT r{};GetClientRect(window,&r);
    if(r.right>0&&r.bottom>0)uiLayout.resize(r.right,r.bottom);
    return uiLayout.logical(GET_X_LPARAM(position),GET_Y_LPARAM(position));
}
void paint(HDC target,int width,int height){
    if(width<=0||height<=0)return;
    const int pixelsWide=width,pixelsHigh=height;
    HDC dc=CreateCompatibleDC(target);HBITMAP bitmap=CreateCompatibleBitmap(target,width,height);auto old=SelectObject(dc,bitmap);
    int saved=SaveDC(dc);uiLayout.resize(width,height);width=uiLayout.width;height=uiLayout.height;
    SetMapMode(dc,MM_ANISOTROPIC);SetWindowExtEx(dc,width,height,nullptr);SetViewportExtEx(dc,pixelsWide,pixelsHigh,nullptr);
    rect(dc,0,0,width,height,RGB(12,18,28));SetBkMode(dc,TRANSPARENT);
    HFONT font=CreateFontW(-18,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Segoe UI");auto oldFont=SelectObject(dc,font);
    auto s=snapshot();Frame frame;{std::lock_guard<std::mutex> lock(shared.mutex);frame=shared.frame;}wizard.buttons.clear();
    text(dc,22,18,L"IKA  /  MICROPHONE + CAMERA SETUP",RGB(80,225,202));
    button(dc,{width-206,12,width-22,48},L"Restart setup",Restart,!wizard.collecting());
    const wchar_t* steps[]={L"1 Devices",L"2 Room noise",L"3 Align",L"4 Verify",L"5 Area",L"6 Test",L"7 Save"};
    int step=(int)wizard.step;for(int i=0;i<7;i++){int x=22+i*(width-44)/7;rect(dc,x,63,(width-60)/7,4,i==step?RGB(80,225,202):i<step?RGB(35,104,99):RGB(40,51,66));text(dc,x,76,steps[i],i==step?RGB(80,225,202):RGB(155,170,189));}
    wrapped(dc,{22,111,width-22,164},instruction());
    int sx=width-354,vy=178,vw=std::max(100,width-402),vh=std::max(100,height-310);double aspect=frame.width?(double)frame.width/frame.height:4./3;
    if((double)vw/vh>aspect)vw=(int)(vh*aspect);else vh=(int)(vw/aspect);wizard.video={22,vy,22+vw,vy+vh};
    rect(dc,22,vy,vw,vh,RGB(25,35,48));
    if(!frame.pixels.empty()){
        BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=frame.width;bi.bmiHeader.biHeight=-frame.height;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;bi.bmiHeader.biCompression=BI_RGB;
        SetStretchBltMode(dc,COLORONCOLOR);StretchDIBits(dc,22,vy,vw,vh,0,0,frame.width,frame.height,frame.pixels.data(),&bi,DIB_RGB_COLORS,SRCCOPY);
        if(s.cameraAge>2)text(dc,36,vy+18,L"CAMERA STALE - check connection",RGB(255,150,95));
    }else {
        text(dc,40,vy+24,L"Waiting for external camera",RGB(255,195,110));
        wrapped(dc,{40,vy+65,22+vw-20,vy+190},s.cameraError.empty()?L"The camera preview will appear here.":L"Camera access failed. Launch Start Setup Wizard.cmd from File Explorer. If needed, enable desktop-app camera access in Windows Settings and close other camera apps.");
    }
    if(wizard.step==Step::Live&&wizard.heatmap&&s.hardwareReady()&&wizard.profileMatches(s)&&wizard.tracker.active&&s.estimate.map.size()==GW*GH){
        const int hw=160,hh=120;BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=hw;bi.bmiHeader.biHeight=-hh;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;
        void* pixels=nullptr;HBITMAP heat=CreateDIBSection(dc,&bi,DIB_RGB_COLORS,&pixels,nullptr,0);HDC hd=CreateCompatibleDC(dc);auto prior=SelectObject(hd,heat);
        double peak=*std::max_element(s.estimate.map.begin(),s.estimate.map.end()),mean=std::accumulate(s.estimate.map.begin(),s.estimate.map.end(),0.)/s.estimate.map.size();auto bytes=(BYTE*)pixels;
        for(int y=0;y<hh;y++)for(int x=0;x<hw;x++){
            auto ray=wizard.profile.projection.unproject({(double)x/(hw-1),(double)y/(hh-1)});int gx=(int)round((ray.x/2.75+1)*.5*(GW-1)),gy=(int)round((1-ray.y/2)*.5*(GH-1));double power=0;
            if(gx>=0&&gx<GW&&gy>=0&&gy<GH)power=std::clamp((s.estimate.map[gy*GW+gx]-mean)/std::max(peak-mean,1e-9),0.,1.);
            int alpha=(int)(pow(power,5)*110),i=(y*hw+x)*4;bytes[i]=0;bytes[i+1]=(BYTE)(alpha*.65);bytes[i+2]=(BYTE)alpha;bytes[i+3]=(BYTE)alpha;
        }
        BLENDFUNCTION blend{AC_SRC_OVER,0,255,AC_SRC_ALPHA};AlphaBlend(dc,22,vy,vw,vh,hd,0,0,hw,hh,blend);SelectObject(hd,prior);DeleteObject(heat);DeleteDC(hd);
    }
    if(step>=4){outline(dc,pixelRect(wizard.profile.region),RGB(80,225,202),2);auto area=pixelRect(wizard.profile.region);text(dc,area.left+8,area.top+8,L"Ika interaction area",RGB(80,225,202));}
    if(wizard.step==Step::Align && wizard.points.size()<4){const setup::Point hints[]={{.22,.5},{.78,.5},{.5,.22},{.5,.78}};crosshair(dc,hints[wizard.points.size()],RGB(255,215,110),true);}
    if((wizard.step==Step::Align||wizard.step==Step::Verify)&&wizard.targetSet)crosshair(dc,wizard.clicked,RGB(255,215,110));
    for(auto p:wizard.points)if(wizard.step==Step::Align)crosshair(dc,p.pixel,RGB(80,225,202),true);
    bool ready=s.hardwareReady()&&wizard.profileMatches(s);
    if(wizard.step==Step::Verify&&wizard.verified&&wizard.alignmentUsable(s))crosshair(dc,wizard.profile.projection.project(wizard.alignmentPoint(s)),RGB(80,225,202));
    if(wizard.step==Step::Live&&ready&&wizard.tracker.active){auto p=wizard.tracker.point;bool inside=wizard.profile.region.contains(p);crosshair(dc,p,inside?RGB(80,255,190):RGB(255,175,80));
        // A region, rather than a pixel-precise point: use held-out setup error as a minimum display radius.
        double radius=std::clamp(wizard.profile.validation+.025,.035,.15);outline(dc,pixelRect({std::clamp(p.x-radius,0.,1.),std::clamp(p.y-radius,0.,1.),std::clamp(p.x+radius,0.,1.),std::clamp(p.y+radius,0.,1.)}),inside?RGB(80,180,150):RGB(210,150,85),1);
        if(p.x<0||p.x>1||p.y<0||p.y>1)text(dc,36,vy+vh-30,L"Strongest sound is outside the camera view",RGB(255,190,100));
    }
    rect(dc,sx,vy,332,height-vy-132,RGB(19,28,41));text(dc,sx+16,vy+14,wizard.step==Step::Live?L"LIVE STATUS":L"SETUP CHECK",RGB(80,225,202));
    text(dc,sx+16,vy+46,s.audioAge<.75&&s.audioError.empty()?L"Array: 16 channels / 48 kHz":L"Array: waiting / unavailable",s.audioAge<.75&&s.audioError.empty()?RGB(195,230,215):RGB(255,165,100));
    text(dc,sx+16,vy+70,s.cameraAge<2?L"Camera: live external camera":L"Camera: waiting / unavailable",s.cameraAge<2?RGB(195,230,215):RGB(255,165,100));
    text(dc,sx+16,vy+98,wizard.toneAlignment()?L"440 Hz level: "+number(s.estimate.tone.level)+L" dBFS":L"Level: "+number(s.estimate.level)+L" dBFS");
    text(dc,sx+16,vy+122,wizard.toneAlignment()?L"Tone clarity: "+number(s.estimate.tone.snr,0)+L" dB":L"Gate: "+number(wizard.profile.gate)+L" dBFS");
    std::wstring detail;
    if(wizard.collecting()){double elapsed=wizard.elapsed();detail=elapsed<0?L"Get ready: "+number(ceil(-elapsed),0):L"Capturing: "+number(std::max(0.,wizard.duration()-elapsed),1)+L" sec left";}
    else if(wizard.step==Step::Align)detail=L"Alignment: "+std::to_wstring(wizard.points.size())+L" / 4 positions";
    else if(wizard.step==Step::Verify)detail=wizard.verified?L"Verification error: "+number(wizard.validation*100)+L"% of frame":L"New position: not verified";
    else if(wizard.step==Step::Noise)detail=wizard.room.good?L"Room floor: "+number(wizard.room.floor)+L" dBFS":L"Room: not measured";
    else if(wizard.step==Step::Live){detail=!ready?L"Unavailable - check devices":!wizard.tracker.active?L"Quiet / uncertain sound":wizard.profile.region.contains(wizard.tracker.point)?L"SOUND IN INTERACTION AREA":L"Sound outside interaction area";}
    else if(wizard.step==Step::Trial)detail=L"Inside: "+std::wstring(wizard.insideDone?L"pass":L"pending")+L"   Outside: "+std::wstring(wizard.outsideDone?L"pass":L"pending");
    else if(wizard.step==Step::Review)detail=L"Alignment verified; location tests passed";
    else detail=L"Check that all microphone bars respond";
    if(wizard.toneAlignment())detail+=wizard.alignmentUsable(s)?L"\n440 Hz: LOCKED":L"\n440 Hz: waiting / unclear";
    wrapped(dc,{sx+16,vy+151,sx+315,vy+213},detail,RGB(80,225,202));
    if(wizard.step==Step::Review){
        text(dc,sx+16,vy+220,L"Room floor: "+number(wizard.profile.noise)+L" dBFS");
        text(dc,sx+16,vy+250,L"Alignment points: 4 / 4");
        text(dc,sx+16,vy+280,L"Independent check: "+number(wizard.profile.validation*100)+L"% error");
        text(dc,sx+16,vy+310,L"Inside sounds in area: "+number(wizard.profile.insideRate*100,0)+L"%");
        text(dc,sx+16,vy+340,L"Outside sounds in area: "+number(wizard.profile.outsideRate*100,0)+L"%");
        text(dc,sx+16,vy+370,L"Device + resolution binding: matched");
    }else for(int i=0;i<16;i++){int col=i/8,row=i%8,x=sx+16+col*158,y=vy+220+row*23;double db=s.audioAge<.75?s.estimate.db[i]:-120;
        text(dc,x,y,std::to_wstring(i+1));rect(dc,x+25,y+5,78,10,RGB(40,51,66));rect(dc,x+25,y+5,(int)(std::clamp((db+90)/70,0.,1.)*78),10,db< -105?RGB(210,80,80):RGB(64,210,170));text(dc,x+108,y,number(db,0));}
    text(dc,sx+16,vy+422,L"Speech detection: NOT ENABLED",RGB(255,195,110));
    text(dc,sx+16,vy+448,L"No audio or video recording");
    if(wizard.step==Step::Review||wizard.step==Step::Live){text(dc,sx+16,vy+474,L"Check error: "+number(wizard.profile.validation*100)+L"% of frame");text(dc,sx+16,vy+498,L"Processing: "+number(s.processingMs,0)+L" ms / block");}
    if(wizard.step==Step::Trial){button(dc,{sx,vy+470,sx+159,vy+510},L"Test inside (8s)",Inside,ready&&!wizard.collecting());button(dc,{sx+170,vy+470,sx+332,vy+510},L"Test outside (8s)",Outside,ready&&!wizard.collecting());}
    if(wizard.step==Step::Devices){button(dc,{sx,vy+470,sx+332,vy+510},L"Retry failed devices",Retry,!wizard.collecting());}
    if(wizard.step==Step::Align||wizard.step==Step::Verify)button(dc,{sx,vy+470,sx+332,vy+510},wizard.phoneTone?L"Source: phone 440 Hz (change)":L"Source: broadband (change)",ToggleSource,!wizard.collecting());
    if(wizard.step==Step::Live){button(dc,{22,height-120,178,height-84},L"More sensitive",MoreSensitive,ready);button(dc,{188,height-120,344,height-84},L"Less sensitive",LessSensitive,ready);button(dc,{354,height-120,530,height-84},L"Retest && save",SaveSensitivity,ready&&wizard.dirty);button(dc,{540,height-120,696,height-84},wizard.heatmap?L"Heatmap: on":L"Heatmap: off",ToggleHeatmap);button(dc,{sx,height-120,sx+332,height-84},L"Retry failed devices",Retry);}
    else if(wizard.step!=Step::Devices)button(dc,{22,height-69,150,height-27},wizard.collecting()?L"Cancel / back":L"Back",Back);
    if(wizard.step!=Step::Live)button(dc,{sx,height-69,sx+332,height-27},wizard.collecting()?L"Capture in progress...":primaryLabel(),Primary,wizard.nextEnabled(s),true);
    int noteX=wizard.step==Step::Live?22:170,noteRight=wizard.step==Step::Live?width-22:sx-18;
    wrapped(dc,{noteX,height-(wizard.step==Step::Live?69:83),noteRight,height-14},wide(wizard.note),RGB(180,198,216));
    if(!s.audioError.empty()&&wizard.step==Step::Devices)wrapped(dc,{40,vy+205,22+vw-20,vy+300},L"Audio: "+wide(s.audioError),RGB(255,165,100));
    SelectObject(dc,oldFont);DeleteObject(font);RestoreDC(dc,saved);
    BitBlt(target,0,0,pixelsWide,pixelsHigh,dc,0,0,SRCCOPY);SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
}

void writeStatus(const Snapshot& s){
    bool calibrated=wizard.step==Step::Live&&wizard.profile.valid()&&wizard.profileMatches(s),ready=calibrated&&s.hardwareReady();
    std::string detected="null",state="setup_required";
    if(calibrated){if(!ready)state="unavailable";else if(wizard.tracker.active){bool inside=wizard.profile.region.contains(wizard.tracker.point);detected=inside?"true":"false";state=inside?"sound_in_region":"sound_outside_region";}else{state=s.estimate.level<wizard.profile.gate?"quiet":"uncertain";if(state=="quiet")detected="false";}}
    auto epoch=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ofstream out("ika-audio-status.json.tmp");out<<std::setprecision(7)<<"{\n  \"schema_version\": 1,\n  \"timestamp_ms\": "<<epoch<<",\n  \"process_id\": "<<GetCurrentProcessId()<<",\n  \"audio_sequence\": "<<s.sequence<<",\n";
    out<<"  \"state\": "<<jsonString(state)<<",\n  \"calibrated\": "<<(calibrated?"true":"false")<<",\n  \"audio_fresh\": "<<(s.audioAge<.75&&s.audioError.empty()?"true":"false")<<",\n  \"camera_fresh\": "<<(s.cameraAge<2&&s.cameraError.empty()?"true":"false")<<",\n";
    out<<"  \"sound_in_interaction_region\": "<<detected<<",\n  \"speech_detected\": null,\n  \"person_speaking\": null,\n  \"speech_detector_enabled\": false,\n";
    out<<"  \"normalized_image_position\": ";if(ready&&wizard.tracker.active)out<<"["<<wizard.tracker.point.x<<", "<<wizard.tracker.point.y<<"]";else out<<"null";
    out<<",\n  \"level_dbfs\": "<<(s.audioAge<.75&&std::isfinite(s.estimate.level)?std::to_string(s.estimate.level):"null")<<",\n  \"gate_dbfs\": "<<wizard.profile.gate<<",\n  \"audio_overruns\": "<<s.overflows<<",\n  \"audio_error\": "<<jsonString(s.audioError)<<",\n  \"camera_error\": "<<jsonString(s.cameraError)<<"\n}\n";
    out.flush();bool ok=bool(out);out.close();if(ok)MoveFileExW(L"ika-audio-status.json.tmp",L"ika-audio-status.json",MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH);
}
setup::Point mousePoint(int x,int y){auto r=wizard.video;return {std::clamp((double)(x-r.left)/std::max(1L,r.right-r.left),0.,1.),std::clamp((double)(y-r.top)/std::max(1L,r.bottom-r.top),0.,1.)};}
LRESULT CALLBACK windowProc(HWND w,UINT m,WPARAM k,LPARAM l){
    if(m==WM_PAINT){PAINTSTRUCT p;HDC dc=BeginPaint(w,&p);RECT r;GetClientRect(w,&r);paint(dc,r.right,r.bottom);EndPaint(w,&p);return 0;}
    if(m==WM_TIMER){auto s=snapshot();wizard.tick(s);static int ticks=0;if(++ticks%5==0)writeStatus(s);InvalidateRect(w,nullptr,FALSE);return 0;}
    if(m==WM_GETMINMAXINFO){auto info=(MINMAXINFO*)l;MONITORINFO monitor{sizeof(MONITORINFO)};GetMonitorInfoW(MonitorFromWindow(w,MONITOR_DEFAULTTONEAREST),&monitor);info->ptMinTrackSize={std::min(960L,monitor.rcWork.right-monitor.rcWork.left),std::min(680L,monitor.rcWork.bottom-monitor.rcWork.top)};return 0;}
    if(m==WM_DPICHANGED){auto r=(RECT*)l;SetWindowPos(w,nullptr,r->left,r->top,r->right-r->left,r->bottom-r->top,SWP_NOZORDER|SWP_NOACTIVATE);InvalidateRect(w,nullptr,FALSE);return 0;}
    if(m==WM_SIZE){InvalidateRect(w,nullptr,FALSE);return 0;}
    if(m==WM_LBUTTONDOWN){POINT p=clientPoint(w,l);int x=p.x,y=p.y;
        if(wizard.step==Step::Zone&&PtInRect(&wizard.video,p)&&!wizard.collecting()){wizard.dragStart=mousePoint(x,y);wizard.dragging=true;SetCapture(w);}return 0;}
    if(m==WM_MOUSEMOVE&&wizard.dragging){auto xy=clientPoint(w,l);auto p=mousePoint(xy.x,xy.y);wizard.profile.region={std::min(p.x,wizard.dragStart.x),std::min(p.y,wizard.dragStart.y),std::max(p.x,wizard.dragStart.x),std::max(p.y,wizard.dragStart.y)};InvalidateRect(w,nullptr,FALSE);return 0;}
    if(m==WM_LBUTTONUP){POINT p=clientPoint(w,l);int x=p.x,y=p.y;
        if(wizard.dragging){wizard.dragging=false;ReleaseCapture();wizard.note=wizard.profile.region.valid()?"Interaction area selected. Continue to test it.":"Draw a larger rectangle (at least 12% of the image in each direction).";return 0;}
        for(auto b:wizard.buttons)if(b.enabled&&PtInRect(&b.bounds,p)){wizard.action(b.action,snapshot());InvalidateRect(w,nullptr,FALSE);return 0;}
        if((wizard.step==Step::Align||wizard.step==Step::Verify)&&!wizard.collecting()&&PtInRect(&wizard.video,p)&&snapshot().hardwareReady()){
            if(wizard.step==Step::Align&&wizard.points.size()==4)return 0;
            wizard.clicked=mousePoint(x,y);wizard.targetSet=true;wizard.verified=false;
            if(wizard.step==Step::Verify)for(auto old:wizard.points)if(hypot(old.pixel.x-wizard.clicked.x,old.pixel.y-wizard.clicked.y)<.12){wizard.targetSet=false;wizard.note="Choose a new point, farther from the four alignment positions.";InvalidateRect(w,nullptr,FALSE);return 0;}
            wizard.note="Source position selected. Click Capture, then keep one sound steady at that position.";InvalidateRect(w,nullptr,FALSE);
        }return 0;
    }
    if(m==WM_KEYDOWN){if(k==VK_ESCAPE)DestroyWindow(w);if(k==VK_RETURN)wizard.primary(snapshot());return 0;}
    if(m==WM_DESTROY){running=false;auto s=snapshot();s.audioAge=s.cameraAge=1e9;s.audioError="Application stopped";s.cameraError="Application stopped";wizard.tracker.reset();writeStatus(s);PostQuitMessage(0);return 0;}
    return DefWindowProcW(w,m,k,l);
}
