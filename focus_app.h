#pragma once
#include <mmsystem.h>
#include <shellapi.h>
#include "focus_dsp.h"
#include "focus_capture.h"

namespace focus {
enum class Recording {Idle,Countdown,Capturing,Saving};
enum Action {Record=101,PlayOriginal,PlayFocused,PlayReference,Stop,OpenFolder,Clear,FovLess,FovMore,Rotate,FlipX,FlipY,Heatmap,Retry};
struct View {
    Geometry geometry;setup::Point mouth{.5,.5};bool selected=false,heatmap=false;
    Recording recording=Recording::Idle;double starts=0,originalDb=-120,focusedDb=-120;size_t samples=0;
    unsigned revision=0,cameraChanges=0;std::wstring directory;std::string targetChangeReason="not_selected";
    int inputGaps=0,queueGaps=0,clippedBlocks=0,queueBaseline=0;
    double captureBegan=0,captureWallSeconds=0;
    std::string alert,lastStopReason;bool alertError=false;
    std::string note="Click your mouth in the camera image. No room measurement or sound alignment is required.";
    bool busy()const{return recording!=Recording::Idle;}
};
struct State {std::mutex mutex;View view;std::vector<float> original,reference,focused;} state;
View view(){std::lock_guard<std::mutex> lock(state.mutex);return state.view;}
bool diagnosticLogging=true;
void eventLocked(const char* event,const std::string& message){
    if(!diagnosticLogging)return;
    auto timestamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ofstream out("focus-events.log",std::ios::app);out<<timestamp<<" pid="<<GetCurrentProcessId()<<" "<<event<<" samples="<<state.view.samples<<" "<<message<<"\n";
}
bool audioAvailable(const Snapshot& s){return s.audioAge<3&&s.audioError.empty()&&!s.audioId.empty()&&s.sequence>0&&std::isfinite(s.estimate.level);}
bool cameraAvailable(const Snapshot& s){return s.cameraAge<2&&s.cameraError.empty()&&!s.cameraId.empty()&&s.width>0&&s.height>0;}
bool abortLocked(const std::string& reason){
    auto& v=state.view;
    if(v.recording!=Recording::Countdown&&v.recording!=Recording::Capturing)return false;
    v.lastStopReason=reason;v.alert=reason;v.alertError=true;v.note=reason;eventLocked("recording_stopped",reason);
    v.recording=Recording::Idle;state.original.clear();state.reference.clear();state.focused.clear();return true;
}
struct CameraBinding {
    int width=0,height=0;std::string id;
    // Caller holds state.mutex. Sound direction/level must never affect the target.
    void updateLocked(const Snapshot& s){
        if(s.width<=0||s.height<=0||s.cameraId.empty())return;
        if(s.width==width&&s.height==height&&s.cameraId==id)return;
        // Freeze the selected ray for the whole clip. Preview changes are not
        // audio failures; apply the new camera geometry after recording/saving.
        if(state.view.busy())return;
        if(width){
            state.view.cameraChanges++;state.view.revision++;
            if(!state.view.busy())state.view.note="Camera geometry changed. Mouth target kept; check its position before recording.";
        }
        width=s.width;height=s.height;id=s.cameraId;
        state.view.geometry.aspect=(double)width/height;
    }
};
void audioFailedLocked(const std::string& error="Audio device failed"){
    bool stopped=abortLocked("Microphone stopped: "+error+". No comparison saved. Target kept; retry devices.");
    state.view.originalDb=state.view.focusedDb=-120;state.view.revision++;
    if(!stopped&&!state.view.busy())state.view.note="Audio unavailable: "+error+". Mouth target kept. Retry devices.";
}
void writeWave(const std::wstring& path,const std::vector<float>& samples,double gain){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot create comparison audio file");
    auto u16=[&](unsigned n){out.put((char)(n&255));out.put((char)((n>>8)&255));};auto u32=[&](unsigned n){u16(n&65535);u16(n>>16);};
    unsigned bytes=(unsigned)samples.size()*2;out.write("RIFF",4);u32(36+bytes);out.write("WAVEfmt ",8);u32(16);u16(1);u16(1);u32((unsigned)FS);u32((unsigned)FS*2);u16(2);u16(16);out.write("data",4);u32(bytes);
    for(float sample:samples){short value=(short)std::lround(std::clamp(sample*gain,-1.,1.)*32767);u16((unsigned short)value);}out.flush();if(!out)throw std::runtime_error("Could not finish writing comparison audio");
}
struct Clip {View settings;Snapshot devices;std::vector<float> original,reference,focused;};
bool collectLocked(const View& processed,const Output& output,const Snapshot& s,Snapshot& recordDevices,Clip& clip,int queueDrops=0,double capturedAt=0){
    auto& v=state.view;
    if(!audioAvailable(s))abortLocked(s.audioError.empty()?"No microphone samples for 3 seconds. No comparison saved. Retry devices.":"Microphone error: "+s.audioError+". No comparison saved. Retry devices.");
    if(capturedAt==0)capturedAt=secondsNow();
    if(v.recording==Recording::Countdown&&capturedAt-BLOCK/FS>=v.starts){v.recording=Recording::Capturing;recordDevices=s;v.queueBaseline=queueDrops;v.captureBegan=capturedAt-BLOCK/FS;eventLocked("recording_started","Collecting ten seconds of received audio");}
    if(v.recording!=Recording::Capturing||processed.revision!=v.revision||output.original.empty())return false;
    v.inputGaps=std::max(0,s.overflows-recordDevices.overflows);v.queueGaps=std::max(0,queueDrops-v.queueBaseline);if(s.estimate.clipping)v.clippedBlocks++;
    if(v.inputGaps||v.queueGaps||v.clippedBlocks){
        if(v.alert.empty())eventLocked("recording_warning","Audio gaps or clipping detected; continuing capture");
        v.alert="Audio gaps: "+std::to_string(v.inputGaps+v.queueGaps)+"; clipped blocks: "+std::to_string(v.clippedBlocks)+". Recording continues. The saved files will include this warning.";v.alertError=false;
    }
    size_t count=std::min(output.original.size(),(size_t)(FS*10)-state.original.size());
    state.original.insert(state.original.end(),output.original.begin(),output.original.begin()+count);state.reference.insert(state.reference.end(),output.reference.begin(),output.reference.begin()+count);state.focused.insert(state.focused.end(),output.focused.begin(),output.focused.begin()+count);v.samples=state.original.size();
    if(v.samples!=(size_t)(FS*10))return false;
    v.captureWallSeconds=std::max(0.,capturedAt-v.captureBegan);clip={v,recordDevices,std::move(state.original),std::move(state.reference),std::move(state.focused)};v.recording=Recording::Saving;v.note="Saving comparison files...";eventLocked("recording_collected",v.alert.empty()?"No audio warnings":v.alert);return true;
}
std::wstring saveClip(const Clip& clip,const wchar_t* root=L"recordings"){
    CreateDirectoryW(root,nullptr);SYSTEMTIME time{};GetLocalTime(&time);wchar_t name[180];
    swprintf_s(name,L"%04u%02u%02u-%02u%02u%02u-%lu-%llu",time.wYear,time.wMonth,time.wDay,time.wHour,time.wMinute,time.wSecond,GetCurrentProcessId(),GetTickCount64());
    std::wstring folder=std::wstring(root)+L"\\"+name;if(!CreateDirectoryW(folder.c_str(),nullptr))throw std::runtime_error("Could not create recording folder");
    double peak=0;for(auto* samples:{&clip.original,&clip.reference,&clip.focused})for(float x:*samples)peak=std::max(peak,std::abs((double)x));
    double gain=peak>1e-9?.9/peak:1.;
    writeWave(folder+L"\\original.wav",clip.original,gain);writeWave(folder+L"\\focused.wav",clip.focused,gain);writeWave(folder+L"\\reference-speech-band.wav",clip.reference,gain);
    auto& g=clip.settings.geometry;auto ray=g.ray(clip.settings.mouth.x,clip.settings.mouth.y);
    std::ofstream metadata(folder+L"\\settings.json");metadata<<std::setprecision(10);
    // Duration counts received samples. PortAudio reports overflow events, not
    // how many samples were lost; do not invent silence or claim continuity.
    metadata<<"{\n  \"mode\": \"geometry_mouth_focus\",\n  \"calibrated\": false,\n  \"speech_detector_enabled\": false,\n  \"sample_rate\": 48000,\n  \"samples\": "<<clip.original.size()<<",\n  \"common_playback_gain\": "<<gain<<",\n";
    metadata<<"  \"duration_basis\": \"received_samples\",\n  \"capture_wall_seconds\": "<<clip.settings.captureWallSeconds<<",\n  \"input_overflow_events\": "<<clip.settings.inputGaps<<",\n  \"processing_queue_dropped_blocks\": "<<clip.settings.queueGaps<<",\n  \"clipped_blocks\": "<<clip.settings.clippedBlocks<<",\n  \"quality_warning\": "<<((clip.settings.inputGaps||clip.settings.queueGaps||clip.settings.clippedBlocks)?"true":"false")<<",\n";
    metadata<<"  \"original\": \"microphone 1; unfiltered; common 32-sample delay\",\n  \"reference\": \"microphone 1; 150-3800 Hz speech band\",\n  \"focused\": \"16-channel fractional delay-and-sum; same 150-3800 Hz speech band\",\n";
    metadata<<"  \"horizontal_fov_degrees\": "<<g.hfov<<",\n  \"image_aspect\": "<<g.aspect<<",\n  \"rotation_quarter_turns\": "<<g.rotation<<",\n  \"flip_x\": "<<(g.flipX?"true":"false")<<",\n  \"flip_y\": "<<(g.flipY?"true":"false")<<",\n";
    metadata<<"  \"mouth_normalized\": ["<<clip.settings.mouth.x<<", "<<clip.settings.mouth.y<<"],\n  \"array_ray_slopes\": ["<<ray.x<<", "<<ray.y<<"],\n  \"audio_id\": "<<jsonString(clip.devices.audioId)<<",\n  \"camera_id\": "<<jsonString(clip.devices.cameraId)<<",\n  \"camera_size\": ["<<clip.devices.width<<", "<<clip.devices.height<<"],\n  \"audio_overruns_at_start\": "<<clip.devices.overflows<<"\n}\n";
    metadata.flush();if(!metadata)throw std::runtime_error("Could not save recording settings");
    wchar_t absolute[MAX_PATH]{};GetFullPathNameW(folder.c_str(),MAX_PATH,absolute,nullptr);return absolute;
}
void audioWorker(){
    CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);std::thread writer;
    while(running){unsigned generation=retryDevices;
        try{BufferedInput input;Beamformer locator;Beam beam;unsigned previousRevision=~0u;int previousOverruns=0,previousQueueDrops=0,markerTick=0;Snapshot recordDevices;CameraBinding camera;AudioPacket packet;
            while(running&&input.next(packet)){auto begin=Clock::now();auto s=snapshot();
                View settings;
                {std::lock_guard<std::mutex> lock(state.mutex);
                    camera.updateLocked(s);
                    settings=state.view;
                }
                auto estimate=settings.heatmap&&!settings.busy()&&++markerTick%4==0?locator.process(packet.samples):levelsOnly(packet.samples);
                if(settings.revision!=previousRevision||packet.overruns!=previousOverruns||packet.queueDrops!=previousQueueDrops)beam.reset();
                previousRevision=settings.revision;previousOverruns=packet.overruns;previousQueueDrops=packet.queueDrops;
                Output output;if(settings.selected)output=beam.process(packet.samples,settings.geometry.ray(settings.mouth.x,settings.mouth.y));
                {std::lock_guard<std::mutex> lock(shared.mutex);shared.estimate=std::move(estimate);shared.audioTime=packet.captured;shared.overflows=packet.overruns;shared.sequence++;shared.processingMs=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
                s=snapshot();Clip clip;bool save=false;
                {std::lock_guard<std::mutex> lock(state.mutex);auto& v=state.view;
                    if(settings.revision==v.revision&&settings.selected){v.originalDb=output.originalDb;v.focusedDb=output.focusedDb;}
                    save=collectLocked(settings,output,s,recordDevices,clip,packet.queueDrops,std::chrono::duration<double>(packet.captured.time_since_epoch()).count());
                }
                if(save){if(writer.joinable())writer.join();writer=std::thread([clip=std::move(clip)](){
                    try{auto path=saveClip(clip);std::lock_guard<std::mutex> lock(state.mutex);auto& v=state.view;v.directory=path;v.recording=Recording::Idle;v.note="Comparison saved. Listen to Original and Focused using the same playback volume.";if(!v.alert.empty())v.alert="Saved with audio warnings: "+std::to_string(v.inputGaps+v.queueGaps)+" gap events, "+std::to_string(v.clippedBlocks)+" clipped blocks. These clips may have clicks or discontinuities; see settings.json.";eventLocked("recording_saved",utf8(path));}
                    catch(const std::exception& e){std::lock_guard<std::mutex> lock(state.mutex);state.view.recording=Recording::Idle;state.view.note=std::string("Save failed: ")+e.what();state.view.alert=state.view.lastStopReason=state.view.note;state.view.alertError=true;eventLocked("save_failed",state.view.note);}
                });}
            }
        }catch(const std::exception& e){{std::lock_guard<std::mutex> lock(shared.mutex);shared.audioError=e.what();}{std::lock_guard<std::mutex> lock(state.mutex);audioFailedLocked(e.what());}}
        if(running)waitForRetry(generation);
    }
    if(writer.joinable())writer.join();CoUninitialize();
}
void action(int id,const Snapshot& s){
    if(id==Stop)PlaySoundW(nullptr,nullptr,0);
    std::unique_lock<std::mutex> lock(state.mutex);auto& v=state.view;
    if(id==Stop){if(v.recording==Recording::Idle)v.note="Playback stopped.";else abortLocked("Recording cancelled. No partial clip was saved.");return;}
    if(v.busy())return;
    if(id==Record){
        if(!v.selected||!audioAvailable(s)){v.note="Select your mouth and wait for microphone audio before recording.";return;}
        lock.unlock();PlaySoundW(nullptr,nullptr,0);lock.lock();state.original.clear();state.reference.clear();state.focused.clear();v.samples=0;v.inputGaps=v.queueGaps=v.clippedBlocks=0;v.captureWallSeconds=0;v.alert.clear();v.alertError=false;v.starts=secondsNow()+3;v.recording=Recording::Countdown;v.note="Keep your mouth at the selected point. Speak for ten seconds after the countdown; normal crowd noise can continue.";eventLocked("recording_requested","Three-second countdown");return;
    }
    if(id==PlayOriginal||id==PlayFocused||id==PlayReference){if(v.directory.empty())return;std::wstring path=v.directory+(id==PlayOriginal?L"\\original.wav":id==PlayFocused?L"\\focused.wav":L"\\reference-speech-band.wav");lock.unlock();bool played=PlaySoundW(path.c_str(),nullptr,SND_FILENAME|SND_ASYNC|SND_NODEFAULT)!=FALSE;lock.lock();v.note=played?"Playing comparison audio. All three clips use the same playback gain.":"Could not play this recording. Open files to play it in your audio player.";return;}
    if(id==OpenFolder){auto path=v.directory;lock.unlock();if(!path.empty())ShellExecuteW(nullptr,L"open",path.c_str(),nullptr,nullptr,SW_SHOWNORMAL);return;}
    if(id==Clear){v.selected=false;v.revision++;v.targetChangeReason="user_cleared";v.note="Target cleared. Click your mouth to choose a new direction.";return;}
    if(id==Heatmap){v.heatmap=!v.heatmap;return;}
    if(id==Retry){retryDevices++;v.note="Retrying failed devices. Close other applications using the array or camera.";return;}
    bool changed=true;
    if(id==FovLess)v.geometry.hfov=std::max(30.,v.geometry.hfov-5);else if(id==FovMore)v.geometry.hfov=std::min(140.,v.geometry.hfov+5);
    else if(id==Rotate)v.geometry.rotation=(v.geometry.rotation+1)%4;else if(id==FlipX)v.geometry.flipX=!v.geometry.flipX;else if(id==FlipY)v.geometry.flipY=!v.geometry.flipY;else changed=false;
    if(changed){v.revision++;v.note="Geometry adjusted. The selected image point now steers the updated direction. Record another comparison to check it.";}
}
void paint(HDC target,int pixelsWide,int pixelsHigh){
    if(pixelsWide<=0||pixelsHigh<=0)return;HDC dc=CreateCompatibleDC(target);HBITMAP bitmap=CreateCompatibleBitmap(target,pixelsWide,pixelsHigh);auto old=SelectObject(dc,bitmap);int saved=SaveDC(dc);
    uiLayout.resize(pixelsWide,pixelsHigh);int width=uiLayout.width,height=uiLayout.height;SetMapMode(dc,MM_ANISOTROPIC);SetWindowExtEx(dc,width,height,nullptr);SetViewportExtEx(dc,pixelsWide,pixelsHigh,nullptr);SetBkMode(dc,TRANSPARENT);
    HFONT font=CreateFontW(-18,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Segoe UI");auto oldFont=SelectObject(dc,font);
    rect(dc,0,0,width,height,RGB(12,18,28));auto s=snapshot();auto v=view();Frame frame;{std::lock_guard<std::mutex> lock(shared.mutex);frame=shared.frame;}wizard.buttons.clear();
    text(dc,22,18,L"IKA / SPEECH FOCUS",RGB(80,225,202));text(dc,22,53,L"Click your mouth, then record a comparison. Keep your mouth at the selected point while speaking.");
    std::wstring geometry=L"Default geometry / uncalibrated   |   Camera FOV: "+number(v.geometry.hfov,0)+L" deg   |   Rotation: "+std::to_wstring(v.geometry.rotation*90)+L" deg   |   Flip X: "+(v.geometry.flipX?L"on":L"off")+L"   Y: "+(v.geometry.flipY?L"on":L"off");text(dc,22,88,geometry,RGB(180,198,216));
    int sx=width-354,vy=140,vw=width-402,vh=height-270;double aspect=frame.width?(double)frame.width/frame.height:4./3.;
    if((double)vw/vh>aspect)vw=(int)(vh*aspect);else vh=(int)(vw/aspect);wizard.video={22,vy,22+vw,vy+vh};rect(dc,22,vy,vw,vh,RGB(25,35,48));
    if(!frame.pixels.empty()){BITMAPINFO bi{};bi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bi.bmiHeader.biWidth=frame.width;bi.bmiHeader.biHeight=-frame.height;bi.bmiHeader.biPlanes=1;bi.bmiHeader.biBitCount=32;bi.bmiHeader.biCompression=BI_RGB;SetStretchBltMode(dc,COLORONCOLOR);StretchDIBits(dc,22,vy,vw,vh,0,0,frame.width,frame.height,frame.pixels.data(),&bi,DIB_RGB_COLORS,SRCCOPY);}
    else wrapped(dc,{40,vy+30,22+vw-25,vy+200},L"Waiting for external camera. Launch Start Speech Focus.cmd from File Explorer. Close other camera apps if needed.");
    if(s.cameraAge>2&&!frame.pixels.empty())text(dc,36,vy+16,L"CAMERA STALE",RGB(255,165,100));
    if(!s.audioError.empty())wrapped(dc,{40,vy+220,22+vw-25,vy+320},L"Audio: "+wide(s.audioError),RGB(255,165,100));
    bool direction=s.hardwareReady()&&s.estimate.level> -85&&s.estimate.contrast>=2.5;
    if(direction&&v.heatmap&&!v.busy()){
        auto p=v.geometry.project({s.estimate.x,s.estimate.y,1});
        if(p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1){crosshair(dc,p,RGB(255,175,80),true);text(dc,36,vy+vh-30,L"Orange: strongest sound (approximate)",RGB(255,195,110));}
    }
    if(v.selected){crosshair(dc,v.mouth,RGB(80,255,210));int x=22+(int)(v.mouth.x*vw),y=vy+(int)(v.mouth.y*vh);text(dc,std::clamp(x+28,30,22+vw-150),std::clamp(y-12,vy+4,vy+vh-24),L"Mouth target",RGB(80,255,210));}
    if(!v.alert.empty()){
        rect(dc,34,vy+12,vw-24,116,v.alertError?RGB(100,30,35):RGB(91,67,22));
        text(dc,46,vy+22,v.alertError?L"RECORDING STOPPED":v.busy()?L"RECORDING CONTINUES - AUDIO WARNING":L"SAVED WITH AUDIO WARNINGS",RGB(255,235,220));
        wrapped(dc,{46,vy+52,22+vw-24,vy+122},wide(v.alert),RGB(255,235,220));
    }
    rect(dc,sx,vy,332,height-vy-112,RGB(19,28,41));text(dc,sx+16,vy+16,L"DIRECTIONAL PICKUP",RGB(80,225,202));
    text(dc,sx+16,vy+49,s.audioAge<.75&&s.audioError.empty()?L"Array: 16 channels / 48 kHz":L"Array: unavailable");text(dc,sx+16,vy+76,s.cameraAge<2&&s.cameraError.empty()?L"Camera: live":L"Camera: unavailable");
    text(dc,sx+16,vy+108,v.selected?L"Mouth target: LOCKED":L"Click your mouth to select it",v.selected?RGB(80,225,202):RGB(255,195,110));
    text(dc,sx+16,vy+137,L"Original: "+(v.selected&&audioAvailable(s)?number(v.originalDb):L"--")+L" dBFS");text(dc,sx+16,vy+163,L"Focused: "+(v.selected&&audioAvailable(s)?number(v.focusedDb):L"--")+L" dBFS");
    std::wstring recordLabel=L"Record comparison (10 sec)";
    if(v.recording==Recording::Countdown)recordLabel=L"Get ready: "+number(std::max(0.,ceil(v.starts-secondsNow())),0);
    if(v.recording==Recording::Capturing)recordLabel=L"Recording: "+number(10-v.samples/FS,1)+L" sec left";
    if(v.recording==Recording::Saving)recordLabel=L"Saving files...";
    button(dc,{sx,vy+198,sx+332,vy+242},recordLabel,Record,!v.busy()&&v.selected&&audioAvailable(s),true);
    if(v.busy()){RECT r{sx,vy+198,sx+332,vy+242};rect(dc,sx,vy+198,332,44,v.recording==Recording::Capturing?RGB(150,48,48):RGB(116,84,27));SetTextColor(dc,RGB(255,245,235));DrawTextW(dc,recordLabel.c_str(),(int)recordLabel.size(),&r,DT_CENTER|DT_VCENTER|DT_SINGLELINE);}
    bool playable=!v.busy()&&!v.directory.empty();
    button(dc,{sx,vy+254,sx+332,vy+292},L"Play original",PlayOriginal,playable);button(dc,{sx,vy+302,sx+332,vy+340},L"Play focused",PlayFocused,playable);
    button(dc,{sx,vy+350,sx+332,vy+388},L"Play speech-band reference",PlayReference,playable);button(dc,{sx,vy+398,sx+332,vy+436},v.busy()?L"Cancel recording":L"Stop playback",Stop,v.recording!=Recording::Saving);
    button(dc,{sx,vy+446,sx+332,vy+484},L"Open comparison files",OpenFolder,playable);button(dc,{sx,vy+494,sx+332,vy+532},L"Clear mouth target",Clear,!v.busy()&&v.selected);
    text(dc,sx+16,vy+550,v.recording==Recording::Capturing?L"Recording audio now (no video)":L"Records audio only when requested",v.recording==Recording::Capturing?RGB(255,150,130):RGB(180,198,216));
    int y=height-100;button(dc,{22,y,102,y+36},L"FOV -",FovLess,!v.busy());button(dc,{110,y,190,y+36},L"FOV +",FovMore,!v.busy());button(dc,{198,y,328,y+36},L"Rotate 90",Rotate,!v.busy());button(dc,{336,y,466,y+36},L"Flip X",FlipX,!v.busy());button(dc,{474,y,604,y+36},L"Flip Y",FlipY,!v.busy());button(dc,{612,y,812,y+36},v.heatmap?L"Sound marker: on":L"Sound marker: off",Heatmap,!v.busy());button(dc,{sx,y,sx+332,y+36},L"Retry failed devices",Retry,!v.busy());
    wrapped(dc,{22,height-53,width-22,height-4},wide(v.note),RGB(180,198,216));
    SelectObject(dc,oldFont);DeleteObject(font);RestoreDC(dc,saved);BitBlt(target,0,0,pixelsWide,pixelsHigh,dc,0,0,SRCCOPY);SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
}
void writeStatus(const Snapshot& s){
    auto v=view();bool ready=s.hardwareReady();auto ray=v.geometry.ray(v.mouth.x,v.mouth.y);auto epoch=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::ofstream out("ika-focus-status.json.tmp");out<<std::setprecision(9)<<"{\n  \"schema_version\": 1,\n  \"mode\": \"geometry_mouth_focus\",\n  \"timestamp_ms\": "<<epoch<<",\n  \"process_id\": "<<GetCurrentProcessId()<<",\n  \"audio_sequence\": "<<s.sequence<<",\n  \"calibrated\": false,\n  \"available\": "<<(ready?"true":"false")<<",\n  \"target_selected\": "<<(v.selected?"true":"false")<<",\n  \"mouth_normalized\": ";
    if(v.selected)out<<"["<<v.mouth.x<<", "<<v.mouth.y<<"]";else out<<"null";
    out<<",\n  \"array_ray_slopes\": ";if(v.selected)out<<"["<<ray.x<<", "<<ray.y<<"]";else out<<"null";
    const char* recording=v.recording==Recording::Idle?"idle":v.recording==Recording::Countdown?"countdown":v.recording==Recording::Capturing?"recording":"saving";
    out<<",\n  \"target_change_reason\": "<<jsonString(v.targetChangeReason)<<",\n  \"camera_change_count\": "<<v.cameraChanges<<",\n  \"recording_state\": "<<jsonString(recording)<<",\n  \"recorded_samples\": "<<v.samples<<",\n  \"status_note\": "<<jsonString(v.note)<<",\n  \"audio_overruns\": "<<s.overflows;
    out<<",\n  \"audio_available\": "<<(audioAvailable(s)?"true":"false")<<",\n  \"recording_alert\": "<<jsonString(v.alert)<<",\n  \"last_stop_reason\": "<<jsonString(v.lastStopReason)<<",\n  \"recording_input_gap_events\": "<<v.inputGaps<<",\n  \"recording_queue_drops\": "<<v.queueGaps<<",\n  \"recording_clipped_blocks\": "<<v.clippedBlocks<<",\n  \"processing_ms\": "<<s.processingMs;
    out<<",\n  \"speech_detected\": null,\n  \"person_speaking\": null,\n  \"speech_detector_enabled\": false,\n  \"audio_error\": "<<jsonString(s.audioError)<<",\n  \"camera_error\": "<<jsonString(s.cameraError)<<"\n}\n";out.flush();bool ok=bool(out);out.close();if(ok)MoveFileExW(L"ika-focus-status.json.tmp",L"ika-focus-status.json",MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH);
}
LRESULT CALLBACK windowProc(HWND window,UINT message,WPARAM key,LPARAM data){
    if(message==WM_PAINT){PAINTSTRUCT ps;HDC dc=BeginPaint(window,&ps);RECT r;GetClientRect(window,&r);focus::paint(dc,r.right,r.bottom);EndPaint(window,&ps);return 0;}
    if(message==WM_TIMER){auto s=snapshot();{std::lock_guard<std::mutex> lock(state.mutex);if(!audioAvailable(s))abortLocked(s.audioError.empty()?"No microphone samples for 3 seconds. No comparison saved. Target kept; retry devices.":"Microphone error: "+s.audioError+". No comparison saved. Target kept; retry devices.");}static int ticks=0;if(++ticks%5==0)focus::writeStatus(s);InvalidateRect(window,nullptr,FALSE);return 0;}
    if(message==WM_LBUTTONUP){auto p=clientPoint(window,data);for(auto b:wizard.buttons)if(b.enabled&&PtInRect(&b.bounds,p)){action(b.action,snapshot());InvalidateRect(window,nullptr,FALSE);return 0;}
        if(PtInRect(&wizard.video,p)&&cameraAvailable(snapshot())){std::lock_guard<std::mutex> lock(state.mutex);if(!state.view.busy()){state.view.mouth=mousePoint(p.x,p.y);state.view.selected=true;state.view.revision++;state.view.targetChangeReason="mouth_clicked";state.view.note="Mouth target locked. Other sounds cannot move it. Click Record comparison to start the ten-second test.";}}InvalidateRect(window,nullptr,FALSE);return 0;}
    if(message==WM_DPICHANGED){auto r=(RECT*)data;SetWindowPos(window,nullptr,r->left,r->top,r->right-r->left,r->bottom-r->top,SWP_NOZORDER|SWP_NOACTIVATE);return 0;}
    if(message==WM_GETMINMAXINFO){auto info=(MINMAXINFO*)data;info->ptMinTrackSize={960,680};return 0;}
    if(message==WM_SIZE){InvalidateRect(window,nullptr,FALSE);return 0;}
    if(message==WM_KEYDOWN){if(key==VK_ESCAPE)DestroyWindow(window);return 0;}
    if(message==WM_DESTROY){running=false;PlaySoundW(nullptr,nullptr,0);auto s=snapshot();s.audioAge=s.cameraAge=1e9;focus::writeStatus(s);PostQuitMessage(0);return 0;}
    return DefWindowProcW(window,message,key,data);
}
int run(){
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);HINSTANCE instance=GetModuleHandle(nullptr);WNDCLASSW wc{};wc.lpfnWndProc=windowProc;wc.hInstance=instance;wc.lpszClassName=L"IkaMicWizard";wc.hCursor=LoadCursor(nullptr,IDC_CROSS);RegisterClassW(&wc);
    RECT work{};SystemParametersInfoW(SPI_GETWORKAREA,0,&work,0);double scale=GetDpiForSystem()/96.;int width=std::min((int)(1360*scale),(int)(work.right-work.left)),height=std::min((int)(940*scale),(int)(work.bottom-work.top));
    HWND window=CreateWindowW(wc.lpszClassName,L"Ika | Mouth-directed Speech Focus",WS_OVERLAPPEDWINDOW,work.left+(work.right-work.left-width)/2,work.top+(work.bottom-work.top-height)/2,width,height,nullptr,nullptr,instance,nullptr);if(!window)throw std::runtime_error("Could not create speech focus window");
    std::thread audio(audioWorker),camera(cameraWorker);ShowWindow(window,SW_SHOW);SetTimer(window,1,66,nullptr);MSG message;while(GetMessageW(&message,nullptr,0,0)>0){TranslateMessage(&message);DispatchMessageW(&message);}running=false;audio.join();camera.join();return 0;
}
}
