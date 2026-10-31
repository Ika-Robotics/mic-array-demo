#pragma once
#include <condition_variable>
#include <deque>

namespace focus {
struct AudioPacket {
    std::vector<float> samples;Clock::time_point captured{};
    int overruns=0,queueDrops=0;
};
class CaptureQueue {
    std::mutex mutex;std::condition_variable available;std::deque<AudioPacket> packets;
    bool finished=false;std::string error;int drops=0;
public:
    void push(AudioPacket packet){
        {std::lock_guard<std::mutex> lock(mutex);if(packets.size()==8){packets.pop_front();drops++;}packets.push_back(std::move(packet));}available.notify_one();
    }
    void finish(const std::string& reason={}){{std::lock_guard<std::mutex> lock(mutex);finished=true;error=reason;}available.notify_all();}
    bool pop(AudioPacket& packet){
        std::unique_lock<std::mutex> lock(mutex);available.wait_for(lock,std::chrono::milliseconds(100),[&]{return !packets.empty()||finished;});
        if(packets.empty()){if(!error.empty())throw std::runtime_error(error);return false;}
        packet=std::move(packets.front());packets.pop_front();packet.queueDrops=drops;return true;
    }
    bool done(){std::lock_guard<std::mutex> lock(mutex);return finished&&packets.empty();}
};
class BufferedInput {
    CaptureQueue queue;std::atomic<bool> stopping{false};std::thread reader;
public:
    BufferedInput():reader([this]{
        CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
        SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL);
        try{
            Audio audio;
            {std::lock_guard<std::mutex> lock(shared.mutex);shared.audioId=audio.hardwareId;shared.audioError.clear();}
            while(running&&!stopping){AudioPacket packet;audio.capture(packet.samples);packet.captured=Clock::now();packet.overruns=audio.overflows;queue.push(std::move(packet));}
            queue.finish();
        }catch(const std::exception& e){queue.finish(e.what());}
        CoUninitialize();
    }){}
    bool next(AudioPacket& packet){while(running&&!stopping){if(queue.pop(packet))return true;if(queue.done())return false;}return false;}
    ~BufferedInput(){stopping=true;if(reader.joinable())reader.join();}
};
inline Estimate levelsOnly(const std::vector<float>& samples){
    Estimate e;if(samples.size()!=BLOCK*CH)throw std::runtime_error("Incomplete microphone block");double total=0;
    for(int c=0;c<CH;c++){double sum=0;for(int t=0;t<BLOCK;t++){double x=samples[t*CH+c];if(!std::isfinite(x))throw std::runtime_error("Invalid microphone samples");sum+=x*x;e.clipping|=std::abs(x)>.99;}e.db[c]=10*log10(std::max(sum/BLOCK,1e-16));total+=sum;}
    e.level=10*log10(std::max(total/(BLOCK*CH),1e-16));return e;
}
}
