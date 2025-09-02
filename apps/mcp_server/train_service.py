import json
from starlette.responses import JSONResponse
from starlette.requests import Request

from packages.predict_lib import train_down_v3
from config.my_paths import DATA_DIR
import pickle

def trigger_train_model(request: Request) -> JSONResponse:
    """Trigger model training with configurable parameters."""
    try:
        # Extract parameters from request query string
        query_params = request.query_params
        token = query_params.get("token")
        period = query_params.get("period", "1y")
        end_date = query_params.get("end_date", "2025-08-10")
        
        # Validate token
        if token != "whoami":
            return JSONResponse(
                {"error": "Unauthorized"},
                status_code=401
            )
        
        # Validate parameters
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            return JSONResponse(
                {"error": f"Invalid period. Must be one of: {', '.join(valid_periods)}"},
                status_code=400
            )
        
        # Trigger training with extracted parameters
        feature_importances, model, report = train_down_v3.main(period=period, end_date_str=end_date)

        feature_importances.to_csv("xgboost_feature_importance.csv", index=False)
        
        # save model as pickle
        with open(DATA_DIR / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        
        # Prepare response with training results
        response_data = {
            "status": "success",
            "parameters": {
                "period": period,
                "end_date": end_date
            },
            "results": {
                "model_trained": True,
                "feature_count": len(feature_importances),
                "top_features": feature_importances.head(10).to_dict(orient="records"),
                "performance": report
            }
        }
        
        # save report as json
        with open(DATA_DIR / 'xgboost_report.json', 'w') as f:
            f.write(json.dumps(response_data))

        
        return JSONResponse(response_data)
        
    except Exception as e:
        return JSONResponse(
            {"error": f"Training failed: {str(e)}"},
            status_code=500
        )

