import os
import time
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


    
def run_talknet(video_path, save_path, in_memory_threshold=0, return_visualization=False, start_seconds=0, end_seconds=-1):
    try:
        from demoTalkNet import setup, main
        import gc
        gc.collect()

        s, DET = setup()
        tracks, scores, files = main(
            s, DET, video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            return_visualization=return_visualization,
            face_boxes="",
            in_memory_threshold=in_memory_threshold,
            save_path=save_path
        )
        
        return tracks, scores, files
    
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")
    

CLIP_DIR = "/home/rhc/licenta4/trimmed_outputs/Integreaza_defectele_in_personaj_Madalina_Dobrovolschi_TEDxICHB_Youth_Live"
MAX_PARALLEL_JOBS = 2  # Set based on your GPU capacity (GTX 1080 can handle 1–2 safely)       

if __name__ == "__main__":
    all_files = sorted(f for f in os.listdir(CLIP_DIR) if f.endswith(".mp4"))
    full_paths = [os.path.join(CLIP_DIR, f) for f in all_files]

    

    os.makedirs("./asd_outputs", exist_ok=True)
    # Get the last part of the path to use as the save directory
    save_paths = ["./asd_outputs/" + os.path.splitext(os.path.basename(path))[0] for path in full_paths]
    for save_path in save_paths:
        os.makedirs(save_path, exist_ok=True)

    #full_paths = [full_paths[10]]
    #save_paths = [save_paths[10]]
    start_time = time.time()
    print(f"Processing {len(full_paths)} video clips in parallel with {MAX_PARALLEL_JOBS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(run_talknet, video_path, save_path, in_memory_threshold=300): video_path
            for video_path, save_path in zip(full_paths, save_paths)
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
            try:
                tracks, scores, files = future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    print(f"\nAll done in {time.time() - start_time:.2f} seconds.")