'@Author: NavinKumarMNK'

import numpy as np
import cv2

class OpticalFlowTracker():
    def __init__ (self):
        self.history = []
        self.tracks = []
        self.frame_idx = 0
        self.min_iou = 0.3
        self.max_age = 100
        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = 1000,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
    
    def start_tracking(self, bboxes):
        for bbox in bboxes:
            x, y, w, h = bbox
            self.tracks.append({
                'bbox': bbox,
                'track_id': len(self.tracks),
                'history': [(x, y)],
                'time_since_update': 0,
                'hits': 1,
                'hit_streak': 1,
                'age': 1,
            })
    
    def update(self, frame, bboxes):
        self.frame_idx += 1
        N = len(self.tracks)
        # Detect bounding boxes in the new frame
        # ...
        M = len(bboxes)

        if N == 0:
            self.start_tracking(bboxes)
        else:
            # Calculate optical flow between the current frame and previous frame
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Associate tracks with new detections using optical flow
            matches = []
            for t, track in enumerate(self.tracks):
                x, y, w, h = track['bbox']
                x1, y1 = int(x + w / 2), int(y + h / 2)
                x2, y2 = int(x1 + flow[y1, x1][0]), int(y1 + flow[y1, x1][1])

                # Check if the optical flow is within the bounding box of a new detection
                for d, det in enumerate(bboxes):
                    if x2 > det[0] and x2 < det[0] + det[2] and y2 > det[1] and y2 < det[1] + det[3]:
                        matches.append((t, d))
                        break

            # Update matched tracks
            for t, d in matches:
                track = self.tracks[t]
                track['bbox'] = bboxes[d]
                track['history'].append(bboxes[d])
                track['time_since_update'] = 0
                track['hits'] += 1
                track['hit_streak'] += 1
                track['age'] += 1

            # Create and initialize new tracks
            unmatched_detections = set(range(M)) - set([d for _, d in matches])
            for d in unmatched_detections:
                self.tracks.append({
                    'bbox': bboxes[d],
                    'track_id': len(self.tracks),
                    'history': [bboxes[d]],
                    'time_since_update': 0,
                    'hits': 1,
                    'hit_streak': 1,
                    'age': 1,
                })

            # Update unmatched tracks
            for t, track in enumerate(self.tracks):
                if t not in [t for t, _ in matches]:
                    track['time_since_update'] += 1
                    track['history'].append(track['bbox'])

            # Remove dead tracks and add it to history object before removing
            for t in reversed(range(len(self.tracks))):
                if self.tracks[t]['time_since_update'] > self.max_age:
                    self.history.append(self.tracks[t])
                    self.tracks.pop(t)

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(self.tracks)
        
    
    def get_history(self):
        return self.history
    
    def get_tracks(self):
        return [track['bbox'] for track in self.tracks], [track['track_id'] for track in self.tracks]


if __name__ == '__main__':
    tracker = OpticalFlowTracker()
    
    # Read video
    cap = cv2.VideoCapture('../../test/street.mp4')
    
    while(cap.isOpened()):
        print("hello")
        ret, frame = cap.read()
        if ret == True:
            # bbox = [x, y, w, h] random for now
            bboxes = [[100, 100, 50, 50], [200, 200, 50, 50]]
            tracker.update(frame, bboxes)
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
