# class TalkNetASD:
#     def __init__(self):
#         from demoTalkNet import setup
#         self.s, self.DET = setup()

#     def __predict__(
#         self,
#         video: str,
#         start_time: float = 0,
#         end_time: float = -1,
#         return_visualization: bool = False,
#         face_boxes: str = "",
#         in_memory_threshold: int = 600,
#     ):
#         """
#         :param video: a video to process
#         :param start_time: the start time of the video to process (in seconds)
#         :param end_time: the end time of the video to process (in seconds). If -1, process until the end of the video.
#         :param return_visualization: whether to return the visualization of the video.
#         :param face_boxes: a string of face boxes in the format "frame_number,x1,y1,x2,y2,x1,y1,x2,y2,..." separated by new lines per frame. If not provided, the model will detect the faces in the video itself to then detect the active speaker.
#         :param in_memory_threshold: the maximum number of frames to load in memory at once. can speed up processing. if 0, this feature is disabled.
#         :return: if return_visualization is True, the first element of the tuple is the output of the model, and the second element is the visualization of the video. Otherwise, the first element is the output of the model.
#         """
#         import gc
#         gc.collect()
#         from demoTalkNet import main
#         def transform_out(out):
#             outputs = []
#             for o in out:
#                 outputs.append({
#                     "frame_number": o['frame_number'],
#                     "boxes": [b for b in o['faces']]
#                 })
#             return outputs
            
#         if return_visualization:
#             out, video_path = main(self.s, self.DET, video, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
#             return video_path
#         else:
#             out = main(self.s, self.DET, video, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
#             return transform_out(out)

# if __name__ == "__main__":
#     TEST_URL = "https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/d979a930-f2a5-4e0d-84fe-a9b233985c4e/dba9cbf3-8374-44bc-8d9d-cc9833d3f502-input-file.mp4"
#     model = TalkNetASD()
#     # change "url" to "path" if you want to test with a local file

#     # get files in \home\rhc\licenta4\trimmed_outputs\Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live 
#     import os, tqdm, time
    
#     files = sorted(os.listdir("/home/rhc/licenta4/trimmed_outputs/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live"))
#     start_time = time.time()
#     count = 0
#     for file in tqdm.tqdm(files, desc="Processing videos", unit="file"):
#         if file.endswith(".mp4"):
            
#             video_path = os.path.join("/home/rhc/licenta4/trimmed_outputs/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live", file)
#             print(f"Processing video: {video_path}")
#             out = model.__predict__(video=video_path, return_visualization=False)

#             if count == 9:
#                 break
#             count += 1

#     end_time = time.time()
#     print(f"TOTAL TIME {end_time-start_time} seconds")
#     #video_path = "/home/rhc/licenta4/video/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live.mp4"
#     #out = model.__predict__(video=video_path, return_visualization=False)
#     #print(list(out))

import os
import json
import time
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

CLIP_DIR = "/home/rhc/licenta4/trimmed_outputs/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live"
MAX_PARALLEL_JOBS = 2  # Set based on your GPU capacity (GTX 1080 can handle 1–2 safely)

def run_talknet_safe(video_path, save_path):
    try:
        return run_talknet(video_path, save_path)
    except Exception as e:
        return video_path, f"Error: {str(e)}"
    
def run_talknet(video_path, save_path):
    from demoTalkNet import setup, main
    import gc
    gc.collect()

    s, DET = setup()
    tracks, scores, files = main(
        s, DET, video_path,
        start_seconds=0,
        end_seconds=-1,
        return_visualization=False,
        face_boxes="",
        in_memory_threshold=300,
        save_path=save_path
    )
    
    best_index = max(range(len(scores)), key=lambda i: np.mean(scores[i]))
    print(f"Best track index: {best_index}, Score: {np.mean(scores[best_index])}")
    return os.path.basename(video_path), len([])


if __name__ == "__main__":
    all_files = sorted(f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4"))
    full_paths = [os.path.join(CLIP_DIR, f) for f in all_files]

    

    os.makedirs("./asd_outputs", exist_ok=True)
    # Get the last part of the path to use as the save directory
    save_paths = ["./asd_outputs/" + os.path.splitext(os.path.basename(path))[0] for path in full_paths]
    for save_path in save_paths:
        os.makedirs(save_path, exist_ok=True)

    full_paths = [full_paths[10]]  # Limit to first 1 for testing
    save_paths = [save_paths[10]]  # Limit to first 1 for testing
    #full_paths = [full_paths[10]]
    #save_paths = [save_paths[10]]
    start_time = time.time()
    print(f"Processing {len(full_paths)} video clips in parallel with {MAX_PARALLEL_JOBS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(run_talknet_safe, video_path, save_path): video_path
            for video_path, save_path in zip(full_paths, save_paths)
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
            try:
                video_name, box_count = future.result()
                print(f"Done: {video_name} → {box_count} frames")
            except Exception as e:
                print(f"Failed: {futures[future]} — {e}")

    print(f"\nAll done in {time.time() - start_time:.2f} seconds.")
