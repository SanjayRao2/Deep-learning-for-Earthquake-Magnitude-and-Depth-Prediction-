# Deep-learning-for-Earthquake-Magnitude-and-Depth-Prediction-
The project aims to predict earthquake magnitudes and depths using deep learning techniques, specifically neural networks and LSTM models.

The Dataset is Procured from https://www.kaggle.com/datasets/usgs/earthquake-database/data

Research Question: How do different neural networks and LSTM models perform in predicting earthquake magnitudes and depths considering input variables such as Latitude, Longitude, Depth and Magnitude?

Description of the Dataset: The output variable, or the target variable, is "Magnitude and Depth". Magnitude is a continuous variable typically ranging from 0 to 10 on the Richter scale, representing the energy released by an earthquake. Depth is also a continuous variable, indicating the distance from the Earth's surface to the hypocenter of the earthquake. It varies from shallow depths, typically less than 70 km, to deep depths, reaching several hundred kilometers. The input to the deep learning model consists of latitude and longitude coordinates, which are continuous variables representing the geographic location of an earthquake event. Latitude ranges from -90° to 90°, where negative values denote the Southern Hemisphere and positive values denote the Northern Hemisphere.
Similarly, longitude ranges from -180° to 180°, where negative values represent the Western Hemisphere and positive values represent the Eastern Hemisphere.


Model Building: In the project, I utilized three different Neural Networks and an LSTM Model.

Neural Network Model 1:
•	Implemented a feedforward neural network architecture with three dense layers. Used a grid search approach to find the optimal combination of hyperparameters, including batch size and epochs.
•	The architecture comprised an input layer with 2 neurons, two hidden layers with 16 neurons each, and an output layer with 2 neurons for magnitude and depth prediction.
•	Activation function: ReLU was used for the hidden layers, and softmax was used for the output layer for classification.
•	Optimizer: Adam optimizer was utilized to minimize the mean squared error loss function.
•	Best parameters obtained from grid search:
o	Batch size: 32
o	Epochs: 20
•	The resulting model achieved a mean squared error (MSE) of -97.94 on the validation data, indicating a relatively low prediction error. This low MSE underscores the model's effectiveness in capturing the complex relationships between earthquake features and their corresponding magnitudes and depths.

Neural Network Model 2:
•	Utilized the KerasRegressor wrapper to integrate Keras models with scikit-learn's GridSearchCV for hyperparameter tuning.
•	The architecture comprised an input layer with 2 neurons, two hidden layers with 16 neurons each, and an output layer with 2 neurons for magnitude and depth prediction.
•	Activation function: ReLU was used for the hidden layers.
•	Optimizer: RMSprop optimizer was utilized for training the model.
•	Best parameters obtained from grid search:
•	Activation function: ReLU
•	Neurons: 16
•	Optimizer: RMSprop
•	The model achieved a mean squared error (MSE) of 0.014 on the validation data, indicating a relatively low prediction error. This low MSE underscores the model's effectiveness in capturing the complex relationships between earthquake features and their corresponding magnitudes and depths.


Neural Network Model 3:
•	Utilized dropout regularization to prevent overfitting, with a dropout rate of 20% after each dense layer.
•	The architecture comprised an input layer with 2 neurons, two hidden layers with 64 neurons each, and an output layer with 2 neurons for magnitude and depth prediction.
•	Activation function: ReLU was used for the hidden layers, and linear activation was used for the output layer for regression.
•	Optimizer: Adam optimizer was utilized to minimize the mean squared error loss function, with a learning rate of 0.001.
•	Trained the model with early stopping to prevent overfitting, with a patience of 100 epochs.
•	The model achieved a mean squared error (MSE) of 5768.42 on the validation data, indicating a relatively low prediction error. This low MSE underscores the model's effectiveness in capturing the complex relationships between earthquake features and their corresponding magnitudes and depths.

LSTM Model: 
•	Implemented a Long Short-Term Memory (LSTM) neural network architecture for sequence prediction.
•	Utilized two LSTM layers with 64 units each, followed by dropout layers to prevent overfitting.
•	Compiled the model using the Adam optimizer and mean squared error loss function.
•	Trained the model for 100 epochs, monitoring the validation loss with early stopping to prevent overfitting.
•	Evaluated the model's performance on the test data, achieving a loss of 4910.75, MAE of 27.23, and MSE of 4910.75.
•	Computed error metrics for both magnitude and depth predictions, obtaining an MAE of 0.31 and MSE of 0.19 for magnitude prediction, and an MAE of 54.15 and MSE of 9821.32 for depth prediction.


Conclusion: In this project, I conducted a comprehensive analysis of Earthquake Data, aiming to predict the magnitude and depth by utilizing deep learning. Deep learning was useful in earthquake prediction due to its ability to handle large and complex datasets effectively. Traditional machine learning approaches may struggle to capture the intricate relationships within such data. Deep learning models, such as neural networks and LSTM networks, excel at learning from raw data and capturing nonlinear dependencies, making them well-suited for earthquake prediction tasks. By leveraging deep learning, we can potentially uncover hidden patterns and trends in seismic data, leading to more accurate earthquake magnitude and depth predictions.
Based on the metrics for the three neural network models and LSTM , the LSTM model generally has the lowest MSE and MAE for both magnitude and depth predictions, indicating better overall performance compared to the other models.
In conclusion, while each model showed improvements over its predecessor to some extent, further refinements in architecture, hyperparameters, and data preprocessing techniques may be necessary to achieve more significant enhancements in performance. 


Lessons Learned: Through this project, I learnt more about: 

•	Architecture Complexity vs. Performance: Increasing the complexity of the neural network architecture and hyperparameters does not always guarantee improved performance. It's essential to strike a balance between model complexity and generalization capability.
•	Data Preprocessing: The choice of data preprocessing techniques, such as scaling, can impact model convergence and performance. 
•	Regularization Techniques: Dropout layers and early stopping were employed to prevent overfitting in the models. Understanding and applying regularization techniques effectively can help improve model generalization.
•	Model Evaluation: Evaluating models on validation and test datasets provides insights into their generalization performance. It's essential to analyze model performance metrics comprehensively to identify areas for improvement.
•	Temporal Dependencies: For sequential data like earthquake records, models like LSTM that can capture temporal dependencies might offer advantages. However, their performance should be compared with simpler architectures to assess their efficacy.


Recommendations: Based on the insights gained from this project, it is recommended to further investigate the impact of additional factors, such as weather conditions and shifts in tectonic plates. Additionally, continuous monitoring and updating of predictive models with new data will enhance their effectiveness in supporting decision-making in the dynamic field of earthquake prediction.
