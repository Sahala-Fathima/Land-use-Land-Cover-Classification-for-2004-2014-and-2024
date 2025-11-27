// Add necessary files 
     //1. {ee.Image} image Input Landsat SR image
     //2. Add ROI 
     //3. Input the necessary LULC class points as Feature collection

// Cloud Mask Function
function cloudMask(image){
  var qa = image.select('QA_PIXEL');
  var dilated = 1 << 1;
  var cirrus = 1 << 2;
  var cloud = 1 << 3;
  var shadow = 1 << 4;
  var mask = qa.bitwiseAnd(dilated).eq(0)
    .and(qa.bitwiseAnd(cirrus).eq(0))
    .and(qa.bitwiseAnd(cloud).eq(0))
    .and(qa.bitwiseAnd(shadow).eq(0));
    
  // Select and rename bands immediately
  var renamed = image.select(
    ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
    ['B1',    'B2',    'B3',    'B4',    'B5',    'B6',    'B7']
  );
  
  return renamed.updateMask(mask)
    .multiply(0.0000275)
    .add(-0.2);
}

// Load and Filter Landsat 9 Data
var imageCollection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2024-01-01', '2024-12-31')
  .map(cloudMask);

// Create Mosaic from 15 Dec 2024
var image1 = l9
  .filterDate('2024-12-15', '2024-12-16')
  .map(cloudMask);

var mosaic = image1.mosaic().clip(roi);

// Create 2024 Median Composite for Filling Gaps
var median2024 = imageCollection.median().clip(roi);


// Fill Gaps in Mosaic using Unmasked Areas from Median Composite
var image = mosaic.unmask(median2024);

print('Bands:', image.bandNames());

// Now you can use `filled` image for classification
Map.centerObject(roi, 10);

// Visualize
Map.addLayer(image, { min: [0.1, 0.05, 0.05], max: [0.4, 0.3, 0.2], bands: ['B5', 'B6', 'B7']}, 'Image');
// Enhanced Visualization Layers for All Classes
var visParams = {
  'True Color': {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3},
  'False Color': {bands: ['B5', 'B4', 'B3'], min: 0, max: 0.4},
  'SWIR Composite': {bands: ['B6', 'B5', 'B4'], min: 0.1, max: 0.4},
  'Urban Composite': {bands: ['B7', 'B6', 'B4'], min: 0.1, max: 0.4}, // Built-up areas
  'Water Composite': {bands: ['B3', 'B5', 'B6'], min: 0, max: 0.3}   // Water-related features
};

for (var name in visParams) {
  Map.addLayer(image, visParams[name], name);
}

// Band map
var bandMap = {
  BLUE: image.select('B2'),
  GREEN: image.select('B3'),
  RED: image.select('B4'),
  NIR: image.select('B5'),
  SWIR1: image.select('B6'),
  SWIR2: image.select('B7')
};

// Add spectral indices
var indices = ee.Image([
  {name: 'NDVI', formula: '(NIR - RED) / (NIR + RED)'},
  {name: 'EVI', formula: '(2.5 * (NIR - RED)) / (NIR + 6 * RED - 7.5 * BLUE + 1)' },
  {name: 'SAVI', formula: '(1.5 * (NIR - RED)) / (NIR + RED + 0.5)'},
  {name: 'MNDWI', formula: '(GREEN - SWIR1) / (GREEN + SWIR1)' },
  {name: 'AWEI', formula: 'BLUE + 2.5*GREEN - 1.5*(NIR + SWIR1) - 0.25*SWIR2'},
  {name: 'NDMI', formula: '(NIR - SWIR1) / (NIR + SWIR1)' },
  {name: 'NDWI', formula: '(GREEN - NIR) / (GREEN + NIR)' },
  {name: 'NDBI', formula: '(SWIR1 - NIR) / (SWIR1 + NIR)' },
  {name: 'NDBaI', formula: '(SWIR1 - SWIR2) / (SWIR1 + SWIR2)' },
  {name: 'UI', formula: '(SWIR2 - NIR) / (SWIR2 + NIR)' },
  {name: 'BSI', formula: '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))'},
  {name: 'GNDVI', formula: '(NIR - GREEN) / (NIR + GREEN)' },
].map(function(dict){
  var indexImage = image.expression(dict.formula, bandMap).rename(dict.name);
  return indexImage;
}));

// Add index & SRTM to image
image = image.addBands(indices).addBands(srtm.clip(roi));

// Variable info
var classValue = [1, 2, 3, 4, 5, 6, 7, 8, 9];
var classNames = ['Built-up', 'Sandy Beach', 'Water', 'Wetland', 'Vegetation', 'Crop Land', 'Barrenland', 'Fallow Land', 'Plantation'];
var classPalette = ['BB3E20', 'F8C859', '294BF8', '008080', '3EC24F', '06E923', '7F4848', '64D6AD', '107E24'];
var columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'EVI', 'SAVI', 'MNDWI', 'AWEI', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'UI', 'BSI', 'GNDVI', 'elevation', 'classvalue', 'sample'];
var features = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'EVI', 'SAVI', 'MNDWI', 'AWEI', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'UI', 'BSI', 'GNDVI', 'elevation'];

// Add Indices Visualization for Specific Classes
Map.addLayer(indices.select('NDVI'), {min: -0.2, max: 0.8}, 'NDVI');
Map.addLayer(indices.select('MNDWI'), {min: -0.5, max: 0.5}, 'MNDWI');
Map.addLayer(indices.select('NDBI'), {min: 0.1, max: 0.4}, 'NDBI');
Map.addLayer(indices.select('BSI'), {min: -0.2, max: 0.4}, 'BSI (Bareland)');
Map.addLayer(indices.select('NDMI'), {min: -0.3, max: 0.3, palette: ['yellow', 'white', 'blue']}, 'NDMI');
Map.addLayer(indices.select('UI'), {min: -0.4, max: 0.3, palette: ['blue', 'white', 'brown']}, 'UI (Urban Index)');
Map.addLayer(indices.select('GNDVI'), {min: -0.2, max: 0.8, palette: ['red', 'white', 'green']}, 'GNDVI');
Map.addLayer(indices.select('SAVI'), {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'SAVI');
Map.addLayer(indices.select('EVI'), {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'EVI');
Map.addLayer(indices.select('AWEI'), {min: -600, max: 600, palette: ['brown', 'white', 'blue']}, 'AWEI (Automated Water Extraction Index)');
Map.addLayer(indices.select('NDBaI'), {min: 0.1, max: 0.3, palette: ['blue', 'white', 'red']}, 'NDBaI');

// Sampels
var samples = built.merge(sandy_beach).merge(water).merge(wetland).merge(vegetation).merge(crop_land).merge(barrenland).merge(fallow_land).merge(plantation)
  .map(function(feat){ return feat.buffer(30) });

// Split samples to train and test per class
samples = ee.FeatureCollection(classValue.map(function(value){
  var features = samples.filter(ee.Filter.eq('classvalue', value)).randomColumn();
  var train = features.filter(ee.Filter.lte('random', 0.8)).map(function(feat){ return feat.set('sample', 'train')});
  var test = features.filter(ee.Filter.gt('random', 0.8)).map(function(feat){ return feat.set('sample', 'test')});
  return train.merge(test);
})).flatten();

// Extract samples
var extract = image.sampleRegions({
  collection: samples,
  scale: 30,
  properties: ['sample', 'classvalue']
});

// Train samples
var train = extract.filter(ee.Filter.eq('sample', 'train'));
print('Train sample size', train.size());
var test = extract.filter(ee.Filter.eq('sample', 'test'));
print('Test sample size', test.size());

// Export image and samples
Export.image.toDrive({
  image: image.toFloat(),
  scale: 30,
  maxPixels: 1e13,
  region: roi,
  crs: 'EPSG:4326',
  folder: 'DL_2',
  description: 'Landsat_2024'
});

Export.table.toDrive({
  collection: extract,
  fileFormat: 'CSV',
  selectors: columns,
  description: 'Samples_LC_SA_2024',
  folder: 'DL_2'
});

// Define hyperparameters to tune
var hyperparameters = {
  numberOfTrees: [100, 200, 300] // Example range for the number of trees
  // You can add more hyperparameters here if needed, e.g.,
  // variablesPerSplit: [null, 5, 10],
  // minLeafPopulation: [1, 5, 10]
};

// Function to train and evaluate the model with given hyperparameters
function trainAndEvaluate(params) {
  var model = ee.Classifier.smileRandomForest(params.numberOfTrees).train(train, 'classvalue', features);
  var cm = test.classify(model, 'predicted').errorMatrix('classvalue', 'predicted');
  return {
    hyperparameters: params,
    accuracy: cm.accuracy(),
    kappa: cm.kappa(),
    model: model // Optionally return the model if you want to inspect it later
  };
}

// Perform hyperparameter tuning (Grid Search in this example)
var tuningResults = [];
for (var i = 0; i < hyperparameters.numberOfTrees.length; i++) {
  var numTrees = hyperparameters.numberOfTrees[i];
  var results = trainAndEvaluate({ numberOfTrees: numTrees });
  tuningResults.push(results);
}

// Find the best model based on accuracy
tuningResults.sort(function(a, b) {
  return b.accuracy.getInfo() - a.accuracy.getInfo(); // Sort in descending order of accuracy
});

var bestResult = tuningResults[0];
print('Best Hyperparameters:', bestResult.hyperparameters);
print('Best Accuracy:', bestResult.accuracy);
print('Best Kappa:', bestResult.kappa);

// Train the final model with the best hyperparameters
var bestModel = ee.Classifier.smileRandomForest(bestResult.hyperparameters.numberOfTrees).train(train, 'classvalue', features);

// Test the best model (optional, as we already evaluated it)
var cmBest = test.classify(bestModel, 'predicted').errorMatrix('classvalue', 'predicted');
print('Confusion matrix of the best model', cmBest, 'Accuracy', cmBest.accuracy(), 'Kappa', cmBest.kappa());

// Apply the best model for classification
var lc = image.classify(bestModel, 'lulc').clip(roi)
  .set('lulc_class_values', classValue, 'lulc_class_palette', classPalette);
Map.addLayer(lc, {}, 'LULC');


// Export the LULC classification result with the best model
Export.image.toDrive({
    image: lc,
    description: 'LULC_Classification_Tuned',
    folder: 'DL_2', // Change the folder name as needed
    scale: 30, // Adjust the spatial resolution (depends on your dataset)
    region: roi, // Define your study area geometry
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
});

// Function to calculate precision, recall, F1-score, and support
function calculateClassificationMetrics(confusionMatrix, classNames) {
  var numClasses = classNames.length;
  var precision = [];
  var recall = [];
  var f1Score = [];
  var support = [];

  var cmArray = confusionMatrix.array().getInfo();

  for (var i = 1; i <= numClasses; i++) {
    var truePositives = cmArray[i][i];
    var falsePositives = 0;
    var falseNegatives = 0;
    var classSupport = 0;

    // Calculate False Positives and Support
    for (var j = 0; j <= numClasses; j++) {
      if (i !== j) {
        falsePositives += cmArray[j][i]; // Sum of column i, excluding diagonal
      }
      classSupport += cmArray[i][j]; // Sum of row i
    }

    // Calculate False Negatives
    for (var j = 0; j <= numClasses; j++) {
      if (i !== j) {
        falseNegatives += cmArray[i][j]; // Sum of row i, excluding diagonal
      }
    }
    support.push(classSupport);

    // Calculate Precision
    var classPrecision = truePositives / (truePositives + falsePositives);
    precision.push(isNaN(classPrecision) ? 0 : classPrecision);

    // Calculate Recall
    var classRecall = truePositives / (truePositives + falseNegatives);
    recall.push(isNaN(classRecall) ? 0 : classRecall);

    // Calculate F1-score
    var classF1Score = 2 * (classPrecision * classRecall) / (classPrecision + classRecall);
    f1Score.push(isNaN(classF1Score) ? 0 : classF1Score);
  }

  // Calculate Macro Average
  var macroAvgPrecision = precision.reduce(function(a, b) { return a + b; }, 0) / numClasses;
  var macroAvgRecall = recall.reduce(function(a, b) { return a + b; }, 0) / numClasses;
  var macroAvgF1Score = f1Score.reduce(function(a, b) { return a + b; }, 0) / numClasses;

  // Calculate Weighted Average
  var totalSupport = support.reduce(function(a, b) { return a + b; }, 0);
  var weightedAvgPrecision = 0;
  var weightedAvgRecall = 0;
  var weightedAvgF1Score = 0;
  for (var i = 0; i < numClasses; i++) {
    weightedAvgPrecision += precision[i] * support[i];
    weightedAvgRecall += recall[i] * support[i];
    weightedAvgF1Score += f1Score[i] * support[i];
  }
  weightedAvgPrecision /= totalSupport;
  weightedAvgRecall /= totalSupport;
  weightedAvgF1Score /= totalSupport;

  return {
    precision: precision,
    recall: recall,
    f1Score: f1Score,
    support: support,
    macroAvg: {
      precision: macroAvgPrecision,
      recall: macroAvgRecall,
      f1Score: macroAvgF1Score
    },
    weightedAvg: {
      precision: weightedAvgPrecision,
      recall: weightedAvgRecall,
      f1Score: weightedAvgF1Score,
      support: totalSupport
    }
  };
}

// Calculate and print the metrics
var metrics = calculateClassificationMetrics(cmBest, classNames);

// Convert to FeatureCollection (FIXED VERSION)
var metricsList = [];

// 1. Add class-wise metrics
for (var i = 0; i < classNames.length; i++) {
  metricsList.push(ee.Feature(null, {
    'Class': classNames[i],
    'Precision': metrics.precision[i],
    'Recall': metrics.recall[i],
    'F1-Score': metrics.f1Score[i],
    'Support': metrics.support[i],
    'Type': 'Class'
  }));
}

// 2. Add averages (FIXED SUPPORT VALUES)
metricsList.push(ee.Feature(null, { // Macro Avg
  'Class': 'Macro Avg',
  'Precision': metrics.macroAvg.precision,
  'Recall': metrics.macroAvg.recall,
  'F1-Score': metrics.macroAvg.f1Score,
  'Support': null, // Changed from '-' to null
  'Type': 'Average'
}));

metricsList.push(ee.Feature(null, { // Weighted Avg
  'Class': 'Weighted Avg',
  'Precision': metrics.weightedAvg.precision,
  'Recall': metrics.weightedAvg.recall,
  'F1-Score': metrics.weightedAvg.f1Score,
  'Support': metrics.weightedAvg.support, // Now correctly referenced
  'Type': 'Average'
}));


// Export corrected collection
Export.table.toDrive({
  collection: ee.FeatureCollection(metricsList),
  description: 'ClassificationMetricsExport',
  fileNamePrefix: 'classification_metrics',
  fileFormat: 'CSV',
  selectors: ['Class', 'Precision', 'Recall', 'F1-Score', 'Support', 'Type']
});

print('Export task created! Check the Tasks tab to run the export.');


var clippedNDVI = indices.select('NDVI').clip(roi)
Export.image.toDrive({
  image: clippedNDVI,
  description: 'NDVI_2024',
  scale: 30,
  region: roi,
  maxPixels: 1e13
});
