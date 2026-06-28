// 1. INTEGRATE SAMPLE POINTS AND ATTRIBUTES
var allPoints = ee.FeatureCollection("projects/dev-fusion-493301-h7/assets/LandslidePoint/Landslide_Point");
var labelProperty = 'kelas';

// 2. LOAD RASTER PARAMETER ASSETS
var soilType = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/SoilType").rename('SoilType');
var landUse = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/LandUse").rename('LandUse');
var flowAcc = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/FlowAcc").rename('FlowAcc');
var aspect = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Aspect").rename('Aspect');
var elevation = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Elevation").rename('Elevation');
var lithology = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Lithology").rename('Lithology');
var rain = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Rain").rename('Rain');
var dist2Road = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Distance2Road").rename('Distance2Road');
var dist2River = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Distance2River").rename('Distance2River');
var slope = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/Slope").rename('Slope');
var slopeLength = ee.Image("projects/dev-fusion-493301-h7/assets/Parameter/SlopeLength").rename('SlopeLength');

var predictors = ee.Image.cat([
  soilType, landUse, flowAcc, aspect, elevation, 
  lithology, rain, dist2Road, dist2River, slope, slopeLength
]);

var bandNames = predictors.bandNames();
var roi = predictors.geometry();

// 3. GENERATE SPATIAL BLOCKS (FOR 10-FOLD CV)
var gridSize = 2500; // Spatial grid of 2.5 km
var proj = ee.Projection('EPSG:4326'); 
var spatialGrid = roi.coveringGrid(proj, gridSize);

var totalFolds = 10; 
var gridWithFolds = spatialGrid.randomColumn('random_val').map(function(feature) {
  var randomFold = ee.Number(feature.get('random_val')).multiply(totalFolds).floor().add(1);
  return feature.set('fold_id', randomFold);
});

var spatialFilter = ee.Filter.intersects({leftField: '.geo', rightField: '.geo'});
var saveFirstJoin = ee.Join.saveFirst({matchKey: 'grid_match'});
var joinedPoints = saveFirstJoin.apply(allPoints, gridWithFolds, spatialFilter);

var pointsWithFolds = joinedPoints.map(function(pt) {
  var gridCell = ee.Feature(pt.get('grid_match'));
  return pt.set('fold_id', gridCell.get('fold_id'));
});

// Extract pixel values to sample points
var sampledPoints = predictors.sampleRegions({
  collection: pointsWithFolds,
  properties: [labelProperty, 'fold_id'],
  scale: 10,
  geometries: true
});

// 4. EVALUATE ACCURACY, COMPILE MATRIX, & EXTRACT PROBABILITIES
var foldIndices = ee.List.sequence(1, totalFolds);

// Collect discrete predictions AND probabilities from 10 folds
var allValidatedSamples = ee.FeatureCollection(foldIndices.map(function(fold) {
  fold = ee.Number(fold);
  var trainSamples = sampledPoints.filter(ee.Filter.neq('fold_id', fold));
  var testSamples = sampledPoints.filter(ee.Filter.eq('fold_id', fold));
  
  // Model A: Discrete class prediction (0 or 1) for Confusion Matrix
  var classifierDiscrete = ee.Classifier.smileRandomForest({
    numberOfTrees: 500, variablesPerSplit: 3, minLeafPopulation: 1, bagFraction: 0.632
  }).train({
    features: trainSamples, classProperty: labelProperty, inputProperties: bandNames
  });
  
  // Model B: Probability prediction (0.00 - 1.00) for ROC-AUC
  var classifierProb = ee.Classifier.smileRandomForest({
    numberOfTrees: 500, variablesPerSplit: 3, minLeafPopulation: 1, bagFraction: 0.632
  }).setOutputMode('PROBABILITY').train({
    features: trainSamples, classProperty: labelProperty, inputProperties: bandNames
  });
  
  // Classify test points
  return testSamples.classify(classifierDiscrete, 'classification')
                    .classify(classifierProb, 'probability');
})).flatten();

// Calculate Global Confusion Matrix
var overallMatrix = allValidatedSamples.errorMatrix(labelProperty, 'classification');

// 5. COMPUTE F1-SCORE & PRINT RESULTS TO GEE CONSOLE
var recallArray = ee.Array(overallMatrix.producersAccuracy()).project([0]); 
var precisionArray = ee.Array(overallMatrix.consumersAccuracy()).project([1]); 

var p_times_r = precisionArray.multiply(recallArray);
var p_plus_r = precisionArray.add(recallArray);
var f1ScoreArray = p_times_r.multiply(2).divide(p_plus_r);

print("=== 10-FOLD SPATIAL CROSS-VALIDATION RESULTS ===");
print("Global Confusion Matrix:", overallMatrix);
print("Overall Accuracy (OA):", overallMatrix.accuracy());
print("Kappa Coefficient:", overallMatrix.kappa());
print("Recall (Producers Accuracy per Class):", overallMatrix.producersAccuracy());
print("Precision (Consumers Accuracy per Class):", overallMatrix.consumersAccuracy());
print("F1-Score (per Class):", f1ScoreArray);

// 6. GENERATE LANDSLIDE SUSCEPTIBILITY MAP & UNCERTAINTY MAP
var finalClassifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 500, variablesPerSplit: 3, minLeafPopulation: 1, bagFraction: 0.632
  })
  .setOutputMode('PROBABILITY') 
  .train({
    features: sampledPoints, classProperty: labelProperty, inputProperties: bandNames
  });

// Enforce GEE to name the output band as 'probability'
var susceptibilityMap = predictors.classify(finalClassifier, 'probability');

// Calculate Shannon Entropy
var prob = susceptibilityMap.select('probability');
var entropy = prob.expression(
  '- (p * log(p) + (1 - p) * log(1 - p))', {
    'p': prob.clamp(0.001, 0.999) // Clamp to avoid log(0)
}).rename('uncertainty');

// 7. VISUALIZATION SETTINGS
var probVisParams = {
  min: 0, max: 1, 
  palette: ['006837', '1a9850', '66bd63', 'a6d96a', 'd9ef8b', 'fee08b', 'fdae61', 'f46d43', 'd73027', 'a50026'] 
};

// Pure Grayscale palette for Shannon Entropy Uncertainty Map
// Value 0 (certainty) is represented by pure white, moving to jet black at maximum entropy (0.693)
var entVisParams = {
  min: 0, 
  max: 0.693, 
  palette: ['#ffffff', '#cccccc', '#969696', '#525252', '#000000'] 
};

Map.centerObject(roi, 11);
Map.setOptions('TERRAIN');
Map.addLayer(susceptibilityMap, probVisParams, 'Landslide Susceptibility Map (Probability)');
Map.addLayer(entropy, entVisParams, 'Uncertainty Map (Shannon Entropy) - Grayscale');

// 8. EXPORT DATA TO GOOGLE DRIVE
// Export Susceptibility Raster Map
Export.image.toDrive({
  image: susceptibilityMap,
  description: 'Landslide_Susceptibility_Map_RF',
  folder: 'GEE_Export_Thesis', 
  scale: 10, 
  region: roi,
  maxPixels: 1e13
});

// Export Uncertainty Raster Map
Export.image.toDrive({
  image: entropy,
  description: 'Uncertainty_Map_Entropy',
  folder: 'GEE_Export_Thesis', 
  scale: 10, 
  region: roi,
  maxPixels: 1e13
});

// Export CSV data for ROC-AUC validation in Python
Export.table.toDrive({
  collection: allValidatedSamples,
  description: 'Spatial_10Fold_CV_Results',
  folder: 'GEE_Export_Thesis',
  fileFormat: 'CSV',
  selectors: [labelProperty, 'classification', 'probability', 'fold_id']
});