import yt_dlp as YT
import os

def download_video(link, Video_path):
    if "shorts/" in link:
        video_id = link.split("shorts/")[-1].split("?")[0]
        link = f"https://www.youtube.com/watch?v={video_id}"
    if os.path.exists(Video_path):
            print("Video already exists:", Video_path)
            return
            
    
    ydl_opts = {'outtmpl': Video_path,'format': 'mp4/bestvideo+bestaudio','quiet': False}
    
    try:
            with YT.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])
            print("Video downloaded successfully with yt-dlp:",Video_path)
    except Exception as e:
            print("Failed to download video:", e)
            
