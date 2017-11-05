# **Finding Lane Lines on the Road** 

## Writeup


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps.   
Step 1. Convert image to grayscale  

Step 2. Apply Gaussian blur  
  Using kernel size of 5.  

Step 3. Apply Canny edge detect  
  The `low_threshold` is 150, and `high_threshold` is 250.

Step 4. Apply image mask  
  The vertices are showed as the following code. I chosed `np.array([[(20,540),(480, 310), (480, 310), (890,540)]], dtype=np.int32)` initially, 
but found it's not robust enough.
```python
vertices = np.array([[(0,img.shape[0]),
                      (int(img.shape[1]/2), int(img.shape[0]*0.6)), 
                      (int(img.shape[1]/2), int(img.shape[0]*0.6)), 
                      (img.shape[1],img.shape[0])]], dtype=np.int32)
```

Step 5. Hough Transform  
The parameters of Hough transform:
```python
    rho = 1
    theta = np.pi/180
    threshold = 15
    min_line_length = 10
    max_line_gap = 50
```
To draw lines more exactly, I defined two funtions to filter and extend lines.  
The first one is `filter_lines()`, which is used to seperate left lane lines and right lane lines, 
and remove the lines with unnormal scopes as noise, judged by the difference with median value.  
The second one, `process_lane_lines()`, call method `filter_lines()`, get the filtered lines.
And then, calculate the top point and bottom point to extend the lines.  
```python
def filter_lines(lines, noise=0.05):
    # filter noise lines
    left_lines, right_lines = np.array([]), np.array([])
    scopes = (lines[:,0,3] - lines[:,0,1])/(lines[:,0,2] - lines[:,0,0])
    left_scopes, right_scopes = scopes[scopes[:]>0], scopes[scopes[:]<0]
    if  left_scopes.size:
        left_mid = np.sort(left_scopes)[left_scopes.size//2]
        left_lines = lines[np.abs(scopes[:]-left_mid) < noise]
    if right_scopes.size:
        right_mid = np.sort(right_scopes)[right_scopes.size//2]
        right_lines = lines[np.abs(scopes[:]-right_mid) < noise]
    if not left_lines.size:
        return right_lines
    if not right_lines.size:
        return left_lines
    return np.vstack((left_lines, right_lines))

def process_lane_lines(img, lines):
    # filter and extend lines
    lines = filter_lines(lines)
    # extend lines
    new_lines = []
    max_y = img.shape[0]
    min_y = min(lines[:, 0, 1].min(), lines[:, 0, 3].min())
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y2 == y1 or x2 == x1:
                continue
            scope = (y2-y1)/(x2-x1)
            if abs(scope) > 1 or abs(scope) < 0.3:
                continue
            # (y2-y1)/(x2-x1) = (min_y-y1)/(min_x-x1)
            min_x = int((min_y-y1)/scope + x1)
            max_x = int((max_y-y1)/scope + x1)
            new_line = [min_x,min_y,max_x,max_y]
            new_lines.append([new_line])
    return np.array(new_lines)
```

Step 6. Combine images
While combining images, I chose a `α` of 0.8, and leave other parameters as default.


### 2. Identify potential shortcomings with your current pipeline

Doesn't effect good enough to process the lane lines around the apex, 
and could be affected by the road condition and the projection of roadside trees.

### 3. Suggest possible improvements to your pipeline

* The lane line segaments could be connected though their endpoints, not just extend to their maximum length.

* The parameters of Candy and Hough Transform need to be changed when there are no left/right lines could be detected, especially lane lines on different pavement, such as bridge check.
