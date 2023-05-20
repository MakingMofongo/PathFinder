# PathFinder
VISTECH - SRM hackathon project

Commits combined to remove extra data

'AI Tech for the Visually Impaired'

<img width="1280" alt="image" src="https://github.com/clumsyninja3086/PathFinder/assets/34381007/9847e144-6e03-4ebe-aaf8-ea1c8d193681">

<img width="1280" alt="image" src="https://github.com/clumsyninja3086/PathFinder/assets/34381007/c23a7d33-55d6-4e67-a458-ab5264460c48">

<img width="1280" alt="image" src="https://github.com/clumsyninja3086/PathFinder/assets/34381007/9b1805a9-a64c-4c9d-8920-3bc6307f36bd">


The problem Vistech solves:

Limited Environmental Awareness: Visually impaired individuals often struggle to perceive and understand their surroundings, leading to a lack of awareness about objects, people, and potential obstacles in their path. Our app's environment description features provide real-time information about the environment, allowing users to better navigate and interact with their surroundings.

Difficulty in Object Identification: Identifying objects without visual cues can be challenging for visually impaired individuals. Our app's object detection and tracking capabilities enable users to recognize and locate objects of interest, empowering them to interact with their environment more effectively.

Navigation Challenges: Moving through unfamiliar or complex environments can be daunting for visually impaired individuals. The path navigation feature in our app offers step-by-step instructions and audio cues, helping users navigate safely and independently, reducing the risk of accidents or getting lost.

Social Interaction Barriers: Recognizing familiar faces and understanding non-verbal cues are essential for meaningful social interactions. The app's face detection and facial inference features assist visually impaired individuals in identifying people they know and understanding their emotions, age, gender, and other attributes, facilitating improved social connections and communication.

Challenges we ran into:

Path Navigation Algorithm: Designing an accurate and reliable path navigation algorithm for visually impaired users was a complex task. We needed to ensure that the algorithm provided step-by-step instructions and audio cues to guide users safely. Overcoming this challenge involved testing various path navigation algorithms, like dijkstra's algorithm, A Star algorithm, etc. We finally settled with the A* algorithm with a custom heuristic function for maintaining a minimum distance between obstacles and the path.

Monocular Depth Map Estimation Accuracy and Processing Speed: Accurate depth estimation from a single camera feed (monocular depth estimation) was crucial for the path navigation feature as we didn't want to use specialized hardware in a product for disadvantaged people. However, achieving high accuracy while maintaining real-time processing speed was a challenging trade-off. We explored various computer vision techniques, optimized algorithms, and utilized parallel processing to improve both accuracy and speed. We settled on a custom implementation of the MIDAS algorithm

Tracking and loss of Tracking Detection: We had to test many tracking algorithms but landed on CV2's KCF tracker for a good balance between speed and accuracy. We encountered challenges when the tracking algorithm lost track of objects due to occlusion, rapid movements, or changes in lighting conditions. To address this, we developed mechanisms to detect and recover from tracking loss. Whenever a track loss is detected, the algorithm switches back to object detection and locates the object of interest and re-establishes a track on it.






<img width="1280" alt="image" src="https://github.com/clumsyninja3086/PathFinder/assets/34381007/879002a2-7c90-4852-aaf1-32cc78759ce9">

