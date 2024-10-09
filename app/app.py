from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import io
import base64
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the serialized model with pipeline (replace with your actual model file)
model = joblib.load('./models/RF-06-10-2024-18-23-00-252516.pkl')

# Define a route for home page with file upload form and result display
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Check if a file is included in the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in the request'}), 400

            file = request.files['file']

            # Check if the file is CSV
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

            # Read CSV into a DataFrame
            df = pd.read_csv(file)

            # Expected columns in the CSV
            expected_columns = ['CustomerId','FirstDay','FirstMonth', 'FirstYear', 'LastDay', 'LastMonth', 'LastYear',
                                'TotalTransactionAmount', 'AverageTransactionAmount','TransactionCount', 'TransactionAmountStdDev', 'MinTransactionAmount',
                                'MaxTransactionAmount', 'Recency', 'Frequency', 'Monetary','Stability']
            print (df.columns)
            # Check if all expected columns are present
            if not all(col in df.columns for col in expected_columns):
                return jsonify({'error': 'Missing required columns in the CSV'}), 400

            # Ensure that the 'Date' column is in the correct datetime format
            # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # # Check for invalid dates
            # if df['Date'].isnull().any():
            #     return jsonify({'error': 'Invalid date format in the "Date" column'}), 400
            
            # df = df.set_index('CustomerId')
            # Preprocess the data and make predictions using the model's pipeline
            predictions = model.predict(df[expected_columns].drop(columns=['CustomerId']))

            # Add predictions to the DataFrame
            df['RiskPrediction'] = predictions
            
            label_counts = df['RiskPrediction'].value_counts()

            fig, ax = plt.subplots(2, 1,figsize=(15, 6))

            sns.countplot(x='RiskPrediction', data=df, ax=ax[0])
            ax[0].set_title('Distribution of Risk')
            ax[0].set_xlabel('Label')
            ax[0].set_ylabel('Count')
            plt.tight_layout()
            
            colors = df['RiskPrediction'].apply(lambda x: 'red' if x == 0 else 'green')
            markers = df['RiskPrediction'].apply(lambda x: 'o' if x == 0 else 'x')
            
            # fig, ax = plt.subplots(2, 1,figsize=(15, 4))
            for i in range(len(df)):
                 ax[1].scatter(df['CustomerId'].iloc[i], df['RiskPrediction'].iloc[i], color=colors.iloc[i], marker=markers.iloc[i])
                 
            # ax.set_title('Predicted Risk of Credit')
            ax[1].set_xlabel('CustomerId')
            ax[1].set_ylabel('Predicted Risk')
            ax[1].grid(True)
            
            red_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class 0 (Low Risk)')
            green_marker = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Class 1 (High Risk)')
            ax[1].legend(handles=[red_marker, green_marker])
            
            ax[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            
            # Show the plots
            # plt.show()



            # Save the plot to a PNG image in memory (bytes)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')

            # Create HTML for displaying the table and the plot
            result_html = '''
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                <title>Credit Risk Prediction API</title>
              </head>
              <body>
                <div style="text-align:center;">
                  <h1>Credit Score Risk Prediction</h1>
                  <p>Upload your CSV file to get sales predictions.</p>
                  <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".csv" required>
                    <button type="submit">Upload and Predict</button>
                  </form>
                  
                  <h3>Credit Risk Prediction (The 1st 10 data)</h3>
                  <table border="1" style="margin: 0 auto; width: 50%;">
                    <tr>
                      <th>Customer Id</th>
                      <th>Credit Risk Prediction</th>
                    </tr>
                    {% for row in predictions %}
                    <tr>
                      <td>{{ row['CustomerId'] }}</td>
                      <td>{{ row['RiskPrediction'] }}</td>
                    </tr>
                    {% endfor %}
                  </table>
                  <h2>Risk Prediction Plot</h2>
                  <img src="data:image/png;base64,{{ plot_url }}" alt="Risk Prediction Plot">
                </div>
              </body>
            </html>
            '''
            
            return render_template_string(
                result_html,
                predictions=df[['CustomerId', 'RiskPrediction']].head(10).to_dict(orient='records'),
                plot_url=plot_url)
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # HTML for the home page
    home_html = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Credit Risk Prediction API</title>
      </head>
      <body>
        <div style="text-align:center;">
          <h1>Welcome to Credit Risk Prediction API</h1>
          <p>Upload your CSV file to get Risk predictions.</p>
          <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Upload and Predict</button>
          </form>
        </div>
      </body>
    </html>
    '''
    return render_template_string(home_html)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
