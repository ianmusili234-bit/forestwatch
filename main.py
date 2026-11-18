import os
import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date

# --------------------------------------------------
# FORESTWATCH AI - CONFIGURATION
# --------------------------------------------------

PROJECT_ID = os.getenv("EE_PROJECT_ID")
AOI_ASSET = os.getenv("AOI_ASSET")
TRAINING_ASSET = os.getenv("TRAINING_ASSET")

# The training asset must contain a property called
# "class": 1 = forest, 0 = non-forest.

app = FastAPI(
    title="ForestWatch AI API",
    description="AI-powered forest monitoring for Mau Forest, Kenya",
    version="1.0"
)

# Allow your classmate's website to connect.
# For deployment, replace "*" with the website domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# GOOGLE EARTH ENGINE CONNECTION
# --------------------------------------------------

def initialize_earth_engine():
    try:
        if not PROJECT_ID:
            raise ValueError("Set the EE_PROJECT_ID environment variable.")

        # Uses an authenticated Google Cloud environment
        # or credentials configured on your computer.
        ee.Initialize(project=PROJECT_ID)

    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
        except Exception as error:
            raise RuntimeError(
                f"Google Earth Engine connection failed: {error}"
            )


# --------------------------------------------------
# IMAGE PREPARATION
# --------------------------------------------------

def mask_clouds(image):
    scl = image.select("SCL")

    # Remove cloud shadows, clouds, cirrus and snow.
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(mask)


def prepare_image(aoi, start_date, end_date):

    images = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_clouds)
    )

    composite = images.median().clip(aoi)

    # Vegetation indicators
    ndvi = composite.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    ndmi = composite.normalizedDifference(
        ["B8", "B11"]
    ).rename("NDMI")

    bands = [
        "B2", "B3", "B4", "B8", "B11", "B12"
    ]

    return composite.select(bands).addBands([ndvi, ndmi])


# --------------------------------------------------
# RANDOM FOREST AI CLASSIFICATION
# --------------------------------------------------

def run_random_forest(aoi, start_date, end_date):

    image = prepare_image(aoi, start_date, end_date)

    training_points = ee.FeatureCollection(TRAINING_ASSET)

    # Sample satellite values at labelled training points.
    samples = image.sampleRegions(
        collection=training_points,
        properties=["class"],
        scale=10,
        geometries=True
    )

    # Split samples into training and testing groups.
    samples = samples.randomColumn("random", seed=42)

    training = samples.filter(
        ee.Filter.lt("random", 0.7)
    )

    testing = samples.filter(
        ee.Filter.gte("random", 0.7)
    )

    # Train the Random Forest model.
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=100,
        seed=42
    ).train(
        features=training,
        classProperty="class",
        inputProperties=image.bandNames()
    )

    # Classify every pixel in the study area.
    classified = image.classify(classifier).rename("class")

    # Test the model using the separate test samples.
    tested = testing.classify(classifier)

    confusion_matrix = tested.errorMatrix(
        "class", "classification"
    )

    accuracy = confusion_matrix.accuracy().getInfo()

    return {
        "image": image,
        "classified": classified,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix.getInfo(),
        "training_samples": training.size().getInfo(),
        "testing_samples": testing.size().getInfo()
    }


# --------------------------------------------------
# FOREST AREA CALCULATION
# --------------------------------------------------

def calculate_class_areas(classified, aoi):

    area_image = (
        ee.Image.pixelArea()
        .divide(10000)
        .addBands(classified)
    )

    result = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=1,
            groupName="class"
        ),
        geometry=aoi,
        scale=10,
        maxPixels=1e13,
        tileScale=4
    ).getInfo()

    areas = {
        "forest_hectares": 0,
        "non_forest_hectares": 0
    }

    for group in result.get("groups", []):
        class_number = int(group["class"])
        hectares = group["sum"]

        if class_number == 1:
            areas["forest_hectares"] = round(hectares, 2)
        elif class_number == 0:
            areas["non_forest_hectares"] = round(hectares, 2)

    return areas


# --------------------------------------------------
# HISTORICAL TREE-COVER LOSS
# --------------------------------------------------

def calculate_historical_loss(aoi):

    hansen = ee.Image(
        "UMD/hansen/global_forest_change_2025_v1_13"
    )

    # Tree cover in the year 2000, using a 30% threshold.
    tree_cover = hansen.select("treecover2000").gte(30)

    # Loss years: 1 = 2001, 25 = 2025.
    loss = hansen.select("lossyear")

    loss_area = (
        ee.Image.pixelArea()
        .updateMask(tree_cover)
        .updateMask(loss.gt(0))
        .divide(10000)
    )

    result = loss_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e13,
        tileScale=4
    ).getInfo()

    return round(result.get("area", 0), 2)


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------

class AnalysisRequest(BaseModel):
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2025, 12, 31)


@app.get("/")
def home():
    return {
        "project": "ForestWatch AI",
        "status": "API is running",
        "study_area": "Mau Forest, Kenya"
    }


@app.post("/analyze")
def analyze_forest(request: AnalysisRequest):

    try:
        if not AOI_ASSET or not TRAINING_ASSET:
            raise ValueError(
                "Set AOI_ASSET and TRAINING_ASSET environment variables."
            )

        if request.start_date >= request.end_date:
            raise ValueError("End date must be after start date.")

        initialize_earth_engine()

        # Load Mau Forest boundary and training samples.
        aoi = ee.FeatureCollection(AOI_ASSET).geometry()

        # Run AI classification.
        model = run_random_forest(
            aoi,
            request.start_date.isoformat(),
            request.end_date.isoformat()
        )

        # Calculate forest and non-forest area.
        areas = calculate_class_areas(
            model["classified"], aoi
        )

        # Calculate historical tree-cover loss.
        historical_loss = calculate_historical_loss(aoi)

        # Create map tile links for the website.
        map_id = model["classified"].getMapId({
            "min": 0,
            "max": 1,
            "palette": ["red", "green"]
        })

        return {
            "project": "ForestWatch AI",
            "study_area": "Mau Forest, Kenya",
            "analysis_period": {
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            },
            "classification": {
                "0": "Non-forest",
                "1": "Forest"
            },
            "forest_area_hectares": areas["forest_hectares"],
            "non_forest_area_hectares": areas["non_forest_hectares"],
            "historical_tree_cover_loss_hectares": historical_loss,
            "random_forest_accuracy": round(model["accuracy"], 4),
            "training_samples": model["training_samples"],
            "testing_samples": model["testing_samples"],
            "confusion_matrix": model["confusion_matrix"],
            "map_tile_url": map_id["tile_fetcher"].url_format
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
