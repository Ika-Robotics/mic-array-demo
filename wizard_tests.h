#pragma once
#include <limits>

void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
setup::Profile testProfile(){
    setup::Profile p;p.audioId="USB\\VID_2752&PID_0019\\TEST";p.cameraId="camera-test";p.width=640;p.height=480;p.noise=-65;p.gate=-58;
    p.validation=.035;p.insideRate=.9;p.outsideRate=.1;p.channels.fill(-65);p.projection.a={.55,.07,.5,.04,-.7,.5};p.projection.residual=.02;return p;
}
void setupTests(){
    using namespace setup;
    for(auto coefficients:std::vector<std::array<double,6>>{{.55,.07,.5,.04,-.7,.5},{0,-.6,.52,.7,0,.47},{-.6,0,.5,0,-.6,.5}}){
        Projection actual;actual.a=coefficients;std::vector<Observation> obs;
        for(Point p:std::vector<Point>{{-.5,0},{.5,0},{0,.5},{0,-.5}})obs.push_back({p,actual.project(p),.01});
        Projection result;std::string error;require(fit(obs,result,error),"Fit rejected a valid rotated/mirrored calibration");
        Point fresh{.17,-.21};auto expected=actual.project(fresh),predicted=result.project(fresh);
        require(hypot(expected.x-predicted.x,expected.y-predicted.y)<1e-8,"Held-out calibration mismatch");
        auto inverse=result.unproject(predicted);require(hypot(inverse.x-fresh.x,inverse.y-fresh.y)<1e-8,"Inverse calibration mismatch");
    }
    Projection result;std::string error;
    std::vector<Observation> degenerate={{{0,0},{.2,.5}},{{0,0},{.8,.5}},{{0,0},{.5,.2}},{{0,0},{.5,.8}}};
    require(!fit(degenerate,result,error),"Degenerate directions accepted");
    auto inconsistent=degenerate;inconsistent[0].direction={-.5,0};inconsistent[1].direction={.5,0};inconsistent[2].direction={0,.5};inconsistent[3].direction={0,-.5};inconsistent[3].pixel={.95,.95};
    require(!fit(inconsistent,result,error),"Inconsistent alignment accepted");
    std::vector<std::array<double,16>> samples(35);for(auto& a:samples)a.fill(-65);std::vector<double> levels(35,-65);
    auto noise=measureNoise(samples,levels,false);require(noise.good&&std::abs(noise.gate+59)<.001,"Quiet-room gate incorrect");
    require(!measureNoise(samples,levels,true).good,"Clipped noise accepted");
    for(int i=0;i<15;i++)levels[i]=-35;require(!measureNoise(samples,levels,false).good,"Changing room noise accepted");levels.assign(35,-65);
    for(auto& a:samples)a[4]=-120;require(!measureNoise(samples,levels,false).good,"Dead microphone accepted");
    std::vector<Point> directions(18,{.25,-.2});auto capture=summarize(directions,22);require(capture.good,"Steady calibration capture rejected");
    require(!summarize(directions,100).good,"Low-quality capture accepted");for(int i=0;i<9;i++)directions[i]={1.5,1.5};require(!summarize(directions,22).good,"Moving source accepted");
    auto p=testProfile();require(p.valid(),"Valid profile rejected");std::stringstream stream;serialize(stream,p);Profile copy;require(deserialize(stream,copy)&&copy.matches(p.audioId,p.cameraId,640,480),"Profile round trip failed");
    require(!copy.matches("other-array",p.cameraId,640,480)&&!copy.matches(p.audioId,p.cameraId,1280,720),"Profile hardware binding failed");
    std::stringstream bad("IKA_MIC_SETUP 1\n\"x\"\n");require(!deserialize(bad,copy),"Truncated profile accepted");
    p.validation=.3;require(!p.valid(),"Unverified profile accepted");p=testProfile();p.outsideRate=.8;require(!p.valid(),"Failed outside test accepted");p=testProfile();p.gate=std::numeric_limits<double>::quiet_NaN();require(!p.valid(),"Nonfinite profile accepted");
    Tracker tracker;tracker.update(true,{.5,.5},1);require(!tracker.active,"One observation immediately detected a source");tracker.update(true,{.51,.5},1.2);require(tracker.active,"Stable source did not activate");tracker.update(false,{},1.6);require(!tracker.active,"Stale detection not cleared");
    Snapshot s;s.audioAge=.1;s.cameraAge=.1;s.sequence=20;s.frames=10;s.width=640;s.height=480;s.audioId="test-array";s.cameraId="test-camera";s.estimate.db.fill(-65);s.estimate.level=-65;
    require(s.hardwareReady(),"Valid hardware check failed");Snapshot stale=s;stale.cameraAge=5;require(!stale.hardwareReady(),"Stale camera accepted");stale=s;stale.estimate.db[3]=-140;require(!stale.hardwareReady(),"Silent channel accepted");
    Wizard w;w.primary(s);require(w.step==Step::Noise,"Devices did not advance to room measurement");w.start(Capture::Noise,s);
    w.began=secondsNow()-4;w.tick(s);require(w.total==0,"Duplicate audio frame counted");s.sequence++;w.tick(s);require(w.total==1,"Fresh audio frame not counted");w.tick(s);require(w.total==1,"Frame counted twice");
    s.audioAge=2;w.tick(s);require(!w.collecting(),"Stale stream did not cancel calibration");
    std::cout<<"PASS: alignment, rotation, reflection, held-out projection, degenerate rejection, room noise, clipping, silent channels, unstable source, profile integrity, device binding, tracking decay, wizard gating and stale capture cancellation"<<std::endl;
    // Exercise the entire wizard with simulated fresh observations, without opening hardware.
    s.audioAge=.1;s.estimate.contrast=3;Wizard flow;flow.phoneTone=false;flow.primary(s);flow.start(Capture::Noise,s);
    auto collect=[&](int frames,Point direction,double level){
        flow.began=secondsNow()-4;
        for(int i=0;i<frames;i++){s.sequence++;s.estimate.x=direction.x;s.estimate.y=direction.y;s.estimate.level=level;s.estimate.db.fill(level);flow.tick(s);}
        flow.began=secondsNow()-flow.duration()-4;flow.tick(s);
    };
    collect(35,{},-65);require(flow.room.good,"Workflow room measurement failed");flow.primary(s);require(flow.step==Step::Align,"Workflow did not enter alignment");
    auto actual=testProfile().projection;
    for(Point point:std::vector<Point>{{-.5,0},{.5,0},{0,.5},{0,-.5}}){flow.clicked=actual.project(point);flow.targetSet=true;flow.primary(s);collect(24,point,-40);}
    require(flow.points.size()==4,"Workflow did not capture four alignment positions");flow.primary(s);require(flow.step==Step::Verify,"Workflow did not enter verification");
    flow.clicked=actual.project({.17,-.21});flow.targetSet=true;flow.primary(s);collect(24,{.17,-.21},-40);require(flow.verified,"Workflow independent verification failed");
    flow.primary(s);require(flow.step==Step::Zone,"Workflow did not enter area selection");flow.primary(s);require(flow.step==Step::Trial,"Workflow did not enter interaction tests");
    flow.action(Inside,s);collect(48,{.1,.1},-40);require(flow.insideDone,"Workflow inside trial failed");flow.action(Outside,s);collect(48,{.65,0},-40);require(flow.outsideDone,"Workflow outside trial failed");
    flow.primary(s);require(flow.step==Step::Review&&flow.profile.valid(),"Workflow produced invalid profile");
    struct DirectoryScope {wchar_t old[MAX_PATH];DirectoryScope(){GetCurrentDirectoryW(MAX_PATH,old);}~DirectoryScope(){SetCurrentDirectoryW(old);}} directoryScope;
    CreateDirectoryW(L"test-output",nullptr);CreateDirectoryW(L"test-output\\workflow",nullptr);require(SetCurrentDirectoryW(L"test-output\\workflow"),"Could not create isolated test directory");
    flow.primary(s);require(flow.step==Step::Live,"Workflow failed to save profile");Wizard restored;restored.load(false);restored.tick(s);require(restored.step==Step::Live,"Saved profile did not automatically load with matching hardware");
    auto originalWizard=wizard;wizard=restored;wizard.tracker.active=true;wizard.tracker.point={.5,.5};writeStatus(s);require(MoveFileExW(L"ika-audio-status.json",L"status-inside.json",MOVEFILE_REPLACE_EXISTING),"Inside JSON not written");
    wizard.tracker.point={.9,.5};writeStatus(s);require(MoveFileExW(L"ika-audio-status.json",L"status-outside.json",MOVEFILE_REPLACE_EXISTING),"Outside JSON not written");
    s.audioAge=2;writeStatus(s);require(MoveFileExW(L"ika-audio-status.json",L"status-stale.json",MOVEFILE_REPLACE_EXISTING),"Stale JSON not written");wizard=originalWizard;
    require(!flow.dirty,"Saved profile still marked dirty");s.audioAge=.1;flow.action(MoreSensitive,s);flow.action(SaveSensitivity,s);require(flow.step==Step::Trial&&!flow.insideDone&&!flow.outsideDone,"Sensitivity changes skipped retesting");
    std::cout<<"PASS: full wizard flow, atomic profile save, automatic profile reload, telemetry fixtures, sensitivity retest"<<std::endl;
    Wizard phone;phone.profile=testProfile();phone.step=Step::Align;phone.targetSet=true;
    s.audioId=phone.profile.audioId;s.cameraId=phone.profile.cameraId;s.width=640;s.height=480;s.estimate.level=-80;s.estimate.contrast=0;
    s.estimate.tone.valid=true;s.estimate.tone.x=.4;s.estimate.tone.y=-.2;
    require(phone.alignmentUsable(s)&&!phone.usable(s),"Tone alignment incorrectly depends on the broadband room gate");
    phone.start(Capture::Align,s);phone.began=secondsNow()-4;
    for(int i=0;i<20;i++){s.sequence++;phone.tick(s);}
    phone.began=secondsNow()-8;phone.tick(s);
    require(phone.points.size()==1&&std::abs(phone.points[0].direction.x-.4)<1e-9,"Phone capture did not use the tone direction");
    s.estimate.tone.valid=false;s.estimate.level=-30;s.estimate.contrast=4;
    phone.targetSet=true;phone.start(Capture::Align,s);phone.began=secondsNow()-4;
    for(int i=0;i<20;i++){s.sequence++;phone.tick(s);}phone.began=secondsNow()-8;phone.tick(s);
    require(phone.points.size()==1,"Broadband crowd sound was accepted without a tone lock");
    phone.step=Step::Verify;phone.verified=true;phone.action(ToggleSource,s);
    require(!phone.phoneTone&&phone.points.empty()&&!phone.verified&&phone.step==Step::Align&&phone.profile.validation==1,"Changing source mode retained incompatible alignment");
    phone.phoneTone=true;phone.step=Step::Trial;
    require(!phone.toneAlignment()&&phone.usable(s),"Phone mode leaked into normal sound trials");
    std::cout<<"PASS: phone-tone capture, independent tone gate, missing-tone rejection, source-switch invalidation and broadband trials"<<std::endl;
}
void saveUiBitmap(const wchar_t* path,int width=1320,int height=900){
    BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=width;bi.bmiHeader.biHeight=-height;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;
    void* data=nullptr;HDC dc=CreateCompatibleDC(nullptr);HBITMAP bitmap=CreateDIBSection(dc,&bi,DIB_RGB_COLORS,&data,nullptr,0);auto old=SelectObject(dc,bitmap);paint(dc,width,height);
    BITMAPFILEHEADER fh{};fh.bfType=0x4D42;fh.bfOffBits=sizeof(fh)+sizeof(BITMAPINFOHEADER);fh.bfSize=fh.bfOffBits+width*height*4;
    std::ofstream out(path,std::ios::binary);out.write((char*)&fh,sizeof(fh));out.write((char*)&bi.bmiHeader,sizeof(BITMAPINFOHEADER));out.write((char*)data,width*height*4);out.close();
    SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
}
void renderWizardTests(){
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    struct DirectoryScope {wchar_t old[MAX_PATH];DirectoryScope(){GetCurrentDirectoryW(MAX_PATH,old);}~DirectoryScope(){SetCurrentDirectoryW(old);}} directoryScope;
    CreateDirectoryW(L"test-output",nullptr);require(SetCurrentDirectoryW(L"test-output"),"Could not open UI test directory");
    shared.frame.width=640;shared.frame.height=480;shared.frame.pixels.resize(640*480*4);
    for(int y=0;y<480;y++)for(int x=0;x<640;x++){int i=(y*640+x)*4;bool grid=x%80<2||y%80<2;shared.frame.pixels[i]=grid?95:45;shared.frame.pixels[i+1]=grid?85:35;shared.frame.pixels[i+2]=grid?70:25;}
    shared.audioTime=shared.cameraTime=Clock::now();shared.frames=20;shared.sequence=30;shared.processingMs=27;shared.estimate.level=-42;shared.estimate.db.fill(-42);shared.estimate.contrast=3;shared.estimate.tone.valid=true;shared.estimate.tone.level=-48;shared.estimate.tone.snr=24;
    auto p=testProfile();shared.audioId=p.audioId;shared.cameraId=p.cameraId;wizard.profile=p;
    const wchar_t* paths[]={L"wizard-01-devices.bmp",L"wizard-02-noise.bmp",L"wizard-03-align.bmp",L"wizard-04-verify.bmp",L"wizard-05-area.bmp",L"wizard-06-test.bmp",L"wizard-07-review.bmp",L"wizard-08-live.bmp"};
    for(int i=0;i<8;i++){
        wizard.step=(Step)i;wizard.note="UI TEST - generated grid, not a camera capture. The physical wizard still needs your measurements.";wizard.targetSet=true;wizard.clicked={.25,.5};wizard.room.good=true;wizard.room.floor=-65;
        wizard.verified=true;wizard.validation=.035;wizard.insideDone=true;wizard.outsideDone=true;wizard.tracker.active=true;wizard.tracker.point={.52,.45};
        saveUiBitmap(paths[i]);
    }
    // Exercise real window messages at fractional and integer display scales.
    // This guards against a readable picture with misaligned calibration clicks.
    WNDCLASSW wc{};wc.lpfnWndProc=windowProc;wc.hInstance=GetModuleHandle(nullptr);wc.lpszClassName=L"IkaWizardLayoutTest";RegisterClassW(&wc);
    HWND window=CreateWindowW(wc.lpszClassName,L"Hidden layout test",WS_POPUP,0,0,1280,840,nullptr,nullptr,wc.hInstance,nullptr);
    require(window!=nullptr,"Could not create hidden layout test window");
    const SIZE sizes[]={{1280,840},{1920,1080},{2880,1800},{3840,2160},{960,680}};
    for(auto size:sizes){
        SetWindowPos(window,nullptr,0,0,size.cx,size.cy,SWP_NOZORDER|SWP_NOACTIVATE);
        wizard.step=Step::Align;wizard.targetSet=false;wizard.points.clear();shared.audioTime=shared.cameraTime=Clock::now();
        auto path=L"wizard-scale-"+std::to_wstring(size.cx)+L"x"+std::to_wstring(size.cy)+L".bmp";saveUiBitmap(path.c_str(),size.cx,size.cy);
        auto position=[&](double x,double y){return MAKELPARAM((int)std::lround(x*uiLayout.scaleX),(int)std::lround(y*uiLayout.scaleY));};
        auto videoPoint=[&](double x,double y){auto r=wizard.video;return position(r.left+x*(r.right-r.left),r.top+y*(r.bottom-r.top));};
        SendMessageW(window,WM_LBUTTONUP,0,videoPoint(.73,.31));
        require(wizard.targetSet&&std::abs(wizard.clicked.x-.73)<.006&&std::abs(wizard.clicked.y-.31)<.006,"Scaled calibration click missed its image position");
        wizard.step=Step::Zone;
        SendMessageW(window,WM_LBUTTONDOWN,MK_LBUTTON,videoPoint(.2,.25));SendMessageW(window,WM_MOUSEMOVE,MK_LBUTTON,videoPoint(.8,.75));SendMessageW(window,WM_LBUTTONUP,0,videoPoint(.8,.75));
        auto r=wizard.profile.region;
        require(!wizard.dragging&&std::abs(r.left-.2)<.006&&std::abs(r.top-.25)<.006&&std::abs(r.right-.8)<.006&&std::abs(r.bottom-.75)<.006,"Scaled interaction-area drag missed its image position");
        // Back is present in the rendered alignment page and returns Zone to Verify.
        bool checked=false;for(auto b:wizard.buttons)if(b.action==Back){SendMessageW(window,WM_LBUTTONUP,0,position((b.bounds.left+b.bounds.right)/2.,(b.bounds.top+b.bounds.bottom)/2.));checked=true;break;}
        require(checked&&wizard.step==Step::Verify,"Scaled Back button missed its target");
    }
    // Avoid production shutdown/telemetry for this synthetic hidden window.
    SetWindowLongPtrW(window,GWLP_WNDPROC,(LONG_PTR)DefWindowProcW);DestroyWindow(window);
    wizard.step=Step::Devices;shared.cameraError="Activate external camera (0x80070005)";shared.frame={};shared.cameraTime={};wizard.note="Camera is unavailable. Setup cannot continue until both devices are live.";saveUiBitmap(L"wizard-09-error.bmp",1180,840);
    std::cout<<"PASS: rendered nine wizard states and five window sizes; calibration clicks, area dragging and buttons work at every scale (no devices opened)."<<std::endl;
}
