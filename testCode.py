import cv2
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def compare_frames(frame1, frame2, threshold):
    diff = cv2.absdiff(frame1, frame2)
    significant_changes = diff > threshold
    return significant_changes

def process_frame_pair(index, frames, n, threshold):
    if index + n < len(frames):
        significant_changes = compare_frames(frames[index], frames[index + n], threshold)
        return index, significant_changes
    return index, None

def highlight_significant_changes(frames, results, width, height, fps, percentage=50):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output.mp4', fourcc, fps, (width, height))

    for index, significant_changes in results:
        result_frame = frames[index].copy()
        if significant_changes is not None:
            mask = significant_changes.astype(np.uint8) * 255
            num_pixels = np.count_nonzero(mask)
            top_pixels = int(num_pixels * (percentage / 100))
            if top_pixels > 0:
                flattened_diffs = mask.reshape(-1)
                most_changed_indices = np.argpartition(flattened_diffs, -top_pixels)[-top_pixels:]
                for idx in most_changed_indices:
                    y, x = divmod(idx, width)
                    cv2.circle(result_frame, (x, y), 1, (0, 0, 255), -1)
        
        out.write(result_frame)
    
    out.release()
    cv2.destroyAllWindows()

def main(num_workers=4):
    video_path = './input.mp4'
    
    n = 2
    threshold = 50
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_frame = {executor.submit(process_frame_pair, i, frames, n, threshold): i for i in range(len(frames) - n)}
        for future in as_completed(future_to_frame):
            index, significant_changes = future.result()
            if significant_changes is not None:
                results.append((index, significant_changes))

    
    highlight_significant_changes(frames, results, width, height, fps)

if __name__ == '__main__':
    main(num_workers=2)


