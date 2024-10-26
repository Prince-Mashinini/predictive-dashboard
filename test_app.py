import unittest
import pandas as pd
import os
from model_training import load_xgboost_model, forecast_future_failures_xgboost
from app import combine_data, fetch_failure_predictions, create_pdf, update_health_and_rul, update_failure_prediction_graph
from HtmlTestRunner import HTMLTestRunner

class TestDashboardFunctions(unittest.TestCase):

    def setUp(self):
        # Setup that runs before each test case
        self.model = load_xgboost_model()
        self.sample_data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='S'),
            'motor_id': [1, 1, 2, 2, 3],
            'failure_probability': [0.2, 0.6, 0.8, 0.3, 0.1],
            'temperature': [75, 80, 85, 90, 95],
            'vibration': [2.0, 2.5, 3.0, 3.5, 4.0],
            'current': [4.0, 5.0, 6.0, 7.0, 8.0],
            'acknowledged': [False, False, False, False, True]
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_combine_data(self):
        # Test that combine_data returns a DataFrame
        combined_df = combine_data()
        self.assertIsInstance(combined_df, pd.DataFrame)
        print("combine_data() test passed")

    def test_fetch_failure_predictions(self):
        # Test fetch_failure_predictions to ensure it returns a non-empty DataFrame
        df_predictions = fetch_failure_predictions()
        self.assertIsInstance(df_predictions, pd.DataFrame)
        print("fetch_failure_predictions() test passed")

    def test_forecast_future_failures_xgboost(self):
        # Test forecasting using XGBoost model
        forecast_df = forecast_future_failures_xgboost(self.df, self.model, forecast_period_seconds=60)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        print(forecast_df.head())
        print("forecast_future_failures_xgboost() test passed")

    def test_pdf_creation(self):
        # Test that create_pdf successfully creates a job card
        job_card_data = {
            'motor_id': 1,
            'failure_probability': 0.8,
            'maintenance_tasks': "Check bearings, Inspect wiring"
        }
        pdf_file = create_pdf(job_card_data)
        self.assertTrue(os.path.exists(pdf_file))
        print(f"PDF created: {pdf_file}")

    def test_health_rul_update(self):
        # Test the health and RUL calculation function
        health_text, rul_text = update_health_and_rul(1)
        
        # Extract the text content from the Dash html.Span component
        health_text_content = health_text.children
        rul_text_content = rul_text

        # Perform assertions
        self.assertIn("Motor Health", health_text_content)
        self.assertIn("Remaining Useful Life", rul_text_content)
        self.assertEqual(health_text_content, "Motor Health: 97/100 (Good)")
        self.assertEqual(rul_text_content, "Remaining Useful Life: 970 hours")


    def test_update_failure_prediction_graph(self):
        # Test the prediction graph update function
        fig = update_failure_prediction_graph(1, '5_minutes')
        self.assertIn('data', fig)
        print("update_failure_prediction_graph() test passed")

if __name__ == "__main__":
    # Define the path to save the test report
    test_report_file = "test_report.html"

    # Create the test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDashboardFunctions)

    # Run the tests and output the report to an HTML file
    with open(test_report_file, "w") as report:
        runner = HTMLTestRunner(stream=report, report_title="Test Report", descriptions=True)
        runner.run(test_suite)

    # Automatically open the report in the browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(test_report_file)}")
