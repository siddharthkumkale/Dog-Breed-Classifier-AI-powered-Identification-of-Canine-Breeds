To create my dog breed prediction model, I used a dataset provided by Stanford University, which contains thousands of images featuring various dog breeds. Each image in the dataset was labeled with the specific breed it represented. This step was crucial because it provided the foundation for teaching my model to recognize and classify different types of dogs.

For building the model itself, I opted for a convolutional neural network (CNN), a type of artificial intelligence architecture known for its effectiveness in image recognition tasks. The CNN was trained using the Stanford Dogs dataset to learn the unique visual features that distinguish one dog breed from another.

Training the model involved feeding it with these labeled images, allowing it to adjust its internal parameters through a process called backpropagation. This iterative learning process enabled the model to improve its accuracy in identifying dog breeds over time.

Once the CNN model was sufficiently trained and achieved good accuracy, I converted it into TensorFlow Lite format. TensorFlow Lite is optimized for mobile and embedded devices, making it suitable for deployment on Android smartphones. This conversion ensured that the model could perform real-time predictions efficiently directly on users' mobile devices.
