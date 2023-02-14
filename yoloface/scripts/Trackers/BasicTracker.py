'@Author: NavinKumarMNK'
import numpy as np

class BasicTracker():
    def __init__ (self):
        self.history = []
        self.tracks = []
        self.frame_idx = 0
        self.min_iou = 0.3
        self.max_age = 100

    
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

    def update(self, bboxes):
        self.frame_idx += 1
        N = len(self.tracks)
        M = len(bboxes)
        if N == 0:
            self.start_tracking(bboxes)
        else:
            iou_matrix = np.zeros((N, M), dtype=np.float32)
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(bboxes):
                    iou_matrix[t, d] = self.iou(track['bbox'], det)
            matched_indices = self.linear_assignment(-iou_matrix)
            unmatched_tracks = []
            for t, track in enumerate(self.tracks):
                if t not in matched_indices[:, 0]:
                    unmatched_tracks.append(t)
            unmatched_detections = []
            for d, det in enumerate(bboxes):
                if d not in matched_indices[:, 1]:
                    unmatched_detections.append(d)
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.min_iou:
                    unmatched_tracks.append(m[0])
                    unmatched_detections.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))
            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)

            for t, track in enumerate(self.tracks):
                if t not in unmatched_tracks:
                    d = matches[np.where(matches[:, 0] == t)[0], 1][0]
                    track['bbox'] = bboxes[d]
                    track['history'].append(bboxes[d])
                    track['time_since_update'] = 0
                    track['hits'] += 1
                    track['hit_streak'] += 1
                    track['age'] += 1

            for i in unmatched_detections:
                self.tracks.append({
                    'bbox': bboxes[i],
                    'track_id': len(self.tracks),
                    'history': [bboxes[i]],
                    'time_since_update': 0,
                    'hits': 1,
                    'hit_streak': 1,
                    'age': 1,
                })
            # update unmatched tracks
            for t in unmatched_tracks:
                track = self.tracks[t]
                track['time_since_update'] += 1
                track['history'].append(track['bbox'])
            
            # remove dead tracks and add it to history object before removing
            for t in reversed(range(len(self.tracks))):
                if self.tracks[t]['time_since_update'] > self.max_age:
                    self.history.append(self.tracks[t])
                    self.tracks.pop(t)


    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (w1 + 1) * (h1 + 1)
        boxBArea = (w2 + 1) * (h2 + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def linear_assignment(self, cost_matrix):
        cost_matrix = cost_matrix.copy()
        assert cost_matrix.ndim == 2
        assert len(cost_matrix) > 0
        assert len(cost_matrix[0]) > 0
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.stack((x, y), axis=-1)
        
    def get_tracks(self):
        return self.tracks

    def get_history(self):
        return self.history

    def get_frame_idx(self):
        return self.frame_idx

    # return newly added tracks and bbox
    def get_new_tracks(self):
        new_tracks = []
        for track in self.tracks:
            if track['age'] == 1:
                new_tracks.append(track)
        return new_tracks
    
    def get_active_tracks(self):
        active_tracks = []
        for track in self.tracks:
            if track['time_since_update'] == 0:
                active_tracks.append(track)
        return active_tracks
    
if __name__ == '__main__':
    tracker = BasicTracker()
    for i in range(10):
        bboxes = np.random.randint(0, 100, size=(10, 4))
        tracker.update(bboxes)
        #print('frame_idx: ', tracker.get_frame_idx())
        print('tracks: ', len(tracker.get_tracks()))
        print('new tracks: ', len(tracker.get_new_tracks()))
    
