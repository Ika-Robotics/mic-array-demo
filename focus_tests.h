#pragma once

std::vector<float> focusSignal(Vec source,int frames=BLOCK*4){
    std::vector<float> audio(frames*CH,0);auto pos=positions();double norm=sqrt(source.x*source.x+source.y*source.y+1);
    for(int c=0;c<CH;c++)for(int f=0;f<31;f++){
        double frequency=350+f*103.27,delay=(pos[c].x*source.x+pos[c].y*source.y)/(norm*SPEED);
        Complex oscillator=std::polar(1.,2*PI*frequency*delay+f*f*.37),step=std::polar(1.,2*PI*frequency/FS);
        for(int t=0;t<frames;t++,oscillator*=step)audio[t*CH+c]+=(float)(.005*oscillator.imag());
    }return audio;
}
double focusEnergy(const std::vector<float>& values){double sum=0;for(size_t i=1024;i<values.size();i++)sum+=(double)values[i]*values[i];return sum;}
void focusTests(){
    using namespace focus;
    diagnosticLogging=false;
    CaptureQueue queue;for(int i=0;i<12;i++){AudioPacket packet;packet.samples={(float)i};packet.overruns=i;queue.push(std::move(packet));}queue.finish();
    for(int i=4;i<12;i++){AudioPacket packet;require(queue.pop(packet)&&packet.samples[0]==i&&packet.queueDrops==4,"Capture queue corrupted ordering or failed to count dropped blocks");}
    require(queue.done(),"Capture queue did not finish");CaptureQueue failed;failed.finish("synthetic read failure");bool failedAsExpected=false;try{AudioPacket packet;failed.pop(packet);}catch(const std::exception& e){failedAsExpected=std::string(e.what())=="synthetic read failure";}require(failedAsExpected,"Capture queue hid the microphone failure reason");
    for(int rotation=0;rotation<4;rotation++)for(bool x:{false,true})for(bool y:{false,true}){
        Geometry geometry;geometry.rotation=rotation;geometry.flipX=x;geometry.flipY=y;
        auto p=geometry.project(geometry.ray(.27,.71));require(hypot(p.x-.27,p.y-.71)<1e-10,"Geometry projection round trip failed");
    }
    Geometry geometry;auto center=geometry.ray(.5,.5);require(center.x==0&&center.y==0,"Default geometry is not centered");
    Vec target{.5,-.25,1};auto speech=focusSignal(target),interference=focusSignal({-1.2,.65,1});
    Beam desiredBeam,noiseBeam;auto desired=desiredBeam.process(speech,target),noise=noiseBeam.process(interference,target);
    double wanted=10*log10(focusEnergy(desired.focused)/focusEnergy(desired.reference)),rejected=10*log10(focusEnergy(noise.focused)/focusEnergy(noise.reference));
    std::cout<<"Target gain vs equally filtered single mic: "<<wanted<<" dB; off-axis gain: "<<rejected<<" dB; SIR improvement: "<<wanted-rejected<<" dB"<<std::endl;
    require(std::abs(wanted)<.3,"Beam lost the selected source");require(wanted-rejected>4,"Insufficient directional rejection in separated-source test");
    Beam split;std::vector<float> streamed;
    for(int block=0;block<4;block++){std::vector<float> chunk(speech.begin()+block*BLOCK*CH,speech.begin()+(block+1)*BLOCK*CH);auto part=split.process(chunk,target);streamed.insert(streamed.end(),part.focused.begin(),part.focused.end());}
    require(streamed==desired.focused,"Audio block boundaries introduced discontinuity");
    Beam quiet;auto silent=quiet.process(std::vector<float>(BLOCK*CH,0),target);require(silent.focusedDb< -100,"Silent input generated focused noise");
    std::mt19937 random(537);std::normal_distribution<double> normal(0,.02);std::vector<float> diffuse(BLOCK*CH*4);for(auto& v:diffuse)v=(float)normal(random);Beam diffuseBeam;auto diffuseOut=diffuseBeam.process(diffuse,target);double diffuseGain=10*log10(focusEnergy(diffuseOut.focused)/focusEnergy(diffuseOut.reference));
    std::cout<<"Independent microphone noise gain: "<<diffuseGain<<" dB"<<std::endl;require(diffuseGain< -10,"Independent microphone noise was not reduced");
    // Same-direction interference is preserved, documenting the separation limit.
    Beam coincident;auto same=coincident.process(speech,target);require(std::abs(10*log10(focusEnergy(same.focused)/focusEnergy(same.reference)))<.3,"Same-direction test contradicted the directional-only model");
    Snapshot s;s.audioAge=s.cameraAge=.1;s.sequence=30;s.frames=10;s.width=640;s.height=480;s.audioId="synthetic-array";s.cameraId="synthetic-camera";s.estimate.db.fill(-40);s.estimate.level=-40;
    state.view=View{};state.view.selected=true;state.view.mouth={.73,.31};CameraBinding camera;camera.updateLocked(s);state.view.recording=Recording::Countdown;state.view.starts=secondsNow()+100;
    Clip clip;Snapshot devices;Output block;block.original.assign(BLOCK,.02f);block.reference.assign(BLOCK,.01f);block.focused.assign(BLOCK,.005f);
    require(!collectLocked(state.view,block,s,devices,clip)&&state.original.empty(),"Recorded during countdown");
    state.view.starts=secondsNow()-1;int completed=0;
    for(int i=0;i<60;i++){
        // A much louder source moves elsewhere; it must not steal the target,
        // reset the beam, cancel the capture or trigger a false camera change.
        s.sequence++;s.estimate.level=-2;s.estimate.db.fill(-2);s.estimate.contrast=10;s.estimate.x=i%2?2.5:-2.5;s.estimate.y=i%3?1.5:-1.5;
        s.overflows=200+i/2;s.estimate.clipping=i%11==0;s.cameraAge=i>=5&&i<=15?4:.1;
        camera.updateLocked(s);if(collectLocked(state.view,block,s,devices,clip,i/10))completed++;
        require(state.view.selected&&state.view.mouth.x==.73&&state.view.mouth.y==.31&&state.view.revision==0,"Louder moving sound or repeated camera updates cleared the mouth target");
    }
    require(completed==1&&clip.original.size()==480000&&clip.focused.size()==480000&&clip.reference.size()==480000,"Recording length or completion was wrong");
    require(clip.settings.inputGaps>0&&clip.settings.queueGaps>0&&clip.settings.clippedBlocks>0&&!clip.settings.alert.empty(),"Recoverable audio defects were not reported on the completed clip");
    auto path=saveClip(clip,L"test-output\\focus-recordings");std::wcout<<L"Synthetic comparison fixture: "<<path<<std::endl;
    auto missing=s;missing.cameraId.clear();missing.width=missing.height=0;camera.updateLocked(missing);
    require(state.view.selected&&state.view.cameraChanges==0,"Missing camera metadata erased the target");
    state.view.recording=Recording::Capturing;s.cameraId="replacement-camera";camera.updateLocked(s);
    require(state.view.selected&&state.view.recording==Recording::Capturing&&state.view.cameraChanges==0,"Camera replacement interrupted fixed-direction recording");state.view.recording=Recording::Idle;camera.updateLocked(s);
    require(state.view.selected&&state.view.mouth.x==.73&&state.view.mouth.y==.31&&state.view.cameraChanges==1,"Deferred camera change lost the target");
    camera.updateLocked(s);require(state.view.cameraChanges==1,"Same camera was repeatedly reported as changed");
    state.view.recording=Recording::Capturing;s.width=1280;s.height=720;camera.updateLocked(s);
    require(state.view.recording==Recording::Capturing&&std::abs(state.view.geometry.aspect-4./3.)<1e-9,"Camera resolution changed the beam during recording");state.view.recording=Recording::Idle;camera.updateLocked(s);
    require(state.view.selected&&std::abs(state.view.geometry.aspect-16./9.)<1e-9,"Deferred camera resolution update failed");
    state.view.recording=Recording::Capturing;audioFailedLocked();
    require(state.view.selected&&state.view.mouth.x==.73&&state.view.mouth.y==.31&&state.view.recording==Recording::Idle,"Audio failure erased the manual target");
    state.view.recording=Recording::Countdown;state.view.starts=secondsNow()-1;s.cameraAge=3;collectLocked(state.view,block,s,devices,clip);require(state.view.recording==Recording::Capturing&&!state.original.empty(),"Stale camera cancelled fixed-direction audio recording");
    s.audioAge=4;collectLocked(state.view,block,s,devices,clip);require(state.view.recording==Recording::Idle&&state.original.empty()&&state.view.alertError&&!state.view.lastStopReason.empty(),"Actual microphone stall did not stop with a visible reason");
    auto stopReason=state.view.alert;state.view.note="Target selected";require(state.view.alert==stopReason,"Routine UI note erased the stop reason");
    state.view.recording=Recording::Capturing;state.original.assign(10,.1f);abortLocked("test cancel");require(state.original.empty()&&state.view.recording==Recording::Idle,"Cancellation retained partial samples");
    require(state.view.selected,"Capture cancellation erased the manual target");
    action(Clear,s);require(!state.view.selected&&state.view.targetChangeReason=="user_cleared","Explicit Clear did not clear the target");
    std::cout<<"PASS: locked mouth survives louder moving sounds through a complete recording, repeated/missing camera metadata, camera changes and audio failure; explicit Clear still works"<<std::endl;
    std::cout<<"PASS: buffered capture ordering/errors, full recording despite repeated overruns/queue drops/clipping/stale video, saved quality flags and persistent microphone-stop explanation"<<std::endl;
    state.view=View{};
    auto start=Clock::now();Beam benchmark;for(int i=0;i<10;i++)benchmark.process(std::vector<float>(speech.begin(),speech.begin()+BLOCK*CH),target);
    std::cout<<"Mean focus processing: "<<std::chrono::duration<double,std::milli>(Clock::now()-start).count()/10<<" ms per 171 ms block"<<std::endl;
    std::cout<<"PASS: geometry transforms, target preservation, directional rejection, streaming continuity, silence, noise, exact capture length, countdown, cancellation and WAV fixtures"<<std::endl;
}
void saveFocusBitmap(const wchar_t* path,int width,int height){
    BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=width;bi.bmiHeader.biHeight=-height;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;
    void* data=nullptr;HDC dc=CreateCompatibleDC(nullptr);HBITMAP bitmap=CreateDIBSection(dc,&bi,DIB_RGB_COLORS,&data,nullptr,0);auto old=SelectObject(dc,bitmap);focus::paint(dc,width,height);
    BITMAPFILEHEADER header{};header.bfType=0x4D42;header.bfOffBits=sizeof(header)+sizeof(BITMAPINFOHEADER);header.bfSize=header.bfOffBits+width*height*4;
    std::ofstream out(path,std::ios::binary);out.write((char*)&header,sizeof(header));out.write((char*)&bi.bmiHeader,sizeof(BITMAPINFOHEADER));out.write((char*)data,width*height*4);
    SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
}
void renderFocusTests(){
    focus::diagnosticLogging=false;
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);CreateDirectoryW(L"test-output",nullptr);
    shared.frame.width=640;shared.frame.height=480;shared.frame.pixels.resize(640*480*4);
    for(int y=0;y<480;y++)for(int x=0;x<640;x++){int i=(y*640+x)*4;bool grid=x%80<2||y%80<2;shared.frame.pixels[i]=grid?95:45;shared.frame.pixels[i+1]=grid?85:35;shared.frame.pixels[i+2]=grid?70:25;}
    shared.audioId="test-array";shared.cameraId="test-camera";shared.audioTime=shared.cameraTime=Clock::now();shared.frames=30;shared.sequence=100;shared.estimate.db.fill(-40);shared.estimate.level=-40;shared.estimate.contrast=3;shared.estimate.x=-.6;shared.estimate.y=.2;
    focus::state.view=focus::View{};saveFocusBitmap(L"test-output\\focus-idle.bmp",1320,900);
    WNDCLASSW wc{};wc.lpfnWndProc=focus::windowProc;wc.hInstance=GetModuleHandle(nullptr);wc.lpszClassName=L"FocusLayoutTest";RegisterClassW(&wc);HWND window=CreateWindowW(wc.lpszClassName,L"Hidden test",WS_POPUP,0,0,1280,840,nullptr,nullptr,wc.hInstance,nullptr);require(window!=nullptr,"Cannot create focus test window");
    for(SIZE size:std::vector<SIZE>{{1280,840},{1920,1080},{2880,1800},{3840,2160},{960,680}}){
        SetWindowPos(window,nullptr,0,0,size.cx,size.cy,SWP_NOZORDER|SWP_NOACTIVATE);shared.audioTime=shared.cameraTime=Clock::now();
        auto file=L"test-output\\focus-"+std::to_wstring(size.cx)+L"x"+std::to_wstring(size.cy)+L".bmp";focus::state.view.selected=true;focus::state.view.mouth={.63,.57};focus::state.view.originalDb=-42;focus::state.view.focusedDb=-48;focus::state.view.note="Mouth target selected. Keep it there and record a comparison. This directs audio pickup; it does not yet confirm who is speaking.";saveFocusBitmap(file.c_str(),size.cx,size.cy);
        auto pos=[&](double x,double y){return MAKELPARAM((int)std::lround(x*uiLayout.scaleX),(int)std::lround(y*uiLayout.scaleY));};auto r=wizard.video;
        SendMessageW(window,WM_LBUTTONUP,0,pos(r.left+.73*(r.right-r.left),r.top+.31*(r.bottom-r.top)));auto v=focus::view();require(v.selected&&hypot(v.mouth.x-.73,v.mouth.y-.31)<.006,"Scaled mouth selection missed its target");
        focus::state.view.recording=focus::Recording::Countdown;SendMessageW(window,WM_LBUTTONUP,0,pos(r.left+.2*(r.right-r.left),r.top+.7*(r.bottom-r.top)));require(focus::view().mouth.x==v.mouth.x,"Mouth moved during recording");focus::state.view.recording=focus::Recording::Idle;
        bool clicked=false;for(auto button:wizard.buttons)if(button.action==focus::Clear){SendMessageW(window,WM_LBUTTONUP,0,pos((button.bounds.left+button.bounds.right)/2.,(button.bounds.top+button.bounds.bottom)/2.));clicked=true;break;}require(clicked&&!focus::view().selected,"Scaled clear-target button missed");
    }
    focus::state.view.selected=true;focus::state.view.recording=focus::Recording::Capturing;focus::state.view.samples=48000*4;focus::state.view.note="Keep your mouth at the selected point. Speak for ten seconds after the countdown; normal crowd noise can continue.";saveFocusBitmap(L"test-output\\focus-recording.bmp",1320,900);
    focus::state.view.recording=focus::Recording::Idle;focus::state.view.directory=L"Synthetic fixture";focus::state.view.note="Comparison saved. Listen to Original and Focused. Speech-band reference separates the effect of frequency filtering from direction filtering.";saveFocusBitmap(L"test-output\\focus-saved.bmp",1320,900);
    focus::state.view.recording=focus::Recording::Capturing;focus::state.view.alert="Audio gaps: 4; clipped blocks: 2. Recording continues. The saved files will include this warning.";focus::state.view.alertError=false;saveFocusBitmap(L"test-output\\focus-warning.bmp",1320,900);
    focus::abortLocked("No microphone samples for 3 seconds. No comparison saved. Target kept; retry devices.");saveFocusBitmap(L"test-output\\focus-stopped.bmp",1320,900);
    SetWindowLongPtrW(window,GWLP_WNDPROC,(LONG_PTR)DefWindowProcW);DestroyWindow(window);
    wchar_t directory[MAX_PATH]{};GetCurrentDirectoryW(MAX_PATH,directory);require(SetCurrentDirectoryW(L"test-output"),"Cannot isolate focus telemetry fixture");focus::writeStatus(snapshot());SetCurrentDirectoryW(directory);
    std::cout<<"PASS: focus UI states, five window sizes, mouth selection and recording lock; no hardware or playback opened"<<std::endl;
}
int focusCaptureCheck(){
    HANDLE singleton=CreateMutexW(nullptr,FALSE,L"Local\\IkaMicArrayWizardV1");DWORD error=GetLastError();
    if(!singleton||error==ERROR_ALREADY_EXISTS){if(singleton)CloseHandle(singleton);std::cout<<"Capture check skipped: close the running microphone demo first."<<std::endl;return 2;}
    try{
        focus::BufferedInput input;focus::Beam beam;focus::AudioPacket packet;int blocks=0,overruns=0,drops=0;double ms=0;auto start=Clock::now();
        while(blocks<64&&input.next(packet)){
            auto began=Clock::now();focus::levelsOnly(packet.samples);beam.process(packet.samples,{0,0,1});ms+=std::chrono::duration<double,std::milli>(Clock::now()-began).count();overruns=packet.overruns;drops=packet.queueDrops;blocks++;
            // Deliberately make the consumer slower than the real focus stage.
            // The reader must continue acquiring while this thread is busy.
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        std::cout<<"Live buffered capture: blocks="<<blocks<<", input_overruns="<<overruns<<", queue_drops="<<drops<<", mean_focus_ms="<<ms/std::max(1,blocks)<<", elapsed_seconds="<<std::chrono::duration<double>(Clock::now()-start).count()<<". No audio saved or played."<<std::endl;
        CloseHandle(singleton);return blocks==64&&overruns==0&&drops==0?0:1;
    }catch(const std::exception& e){CloseHandle(singleton);std::cout<<"Live capture check unavailable: "<<e.what()<<std::endl;return 2;}
}
