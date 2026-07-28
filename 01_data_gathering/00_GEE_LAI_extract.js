// Define the study area）
var region = ee.Geometry.Polygon([
  [[-31.5, 71.2], [41.5, 71.2], [41.5, 34.5], [-31.5, 34.5], [-31.5, 71.2]]
]);

// Load and merge Terra Aqua data
var terraLAI = ee.ImageCollection("MODIS/061/MOD15A2H")
                .filterDate('2000-02-18', '2023-12-31') 
                .filterBounds(region)
                .select('Lai_500m')
                .map(function(image) {
                  return image.multiply(0.1) 
                              .copyProperties(image, ['system:time_start']);
                });

var aquaLAI = ee.ImageCollection("MODIS/061/MYD15A2H")
                .filterDate('2002-07-04', '2024-12-31') 
                .filterBounds(region)
                .select('Lai_500m')
                .map(function(image) {
                  return image.multiply(0.1) 
                              .copyProperties(image, ['system:time_start']);
                });

var combinedLAI = aquaLAI.merge(terraLAI)
                         .sort('system:time_start')
                         .map(function(image) {
                           return image.updateMask(image.gte(0).and(image.lte(10))); 
                         });

// forest mask
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var forestMask = gfc.select('treecover2000').gte(25); 

// reproject and clip
var reprojectedLAI = combinedLAI.map(function(image) {
  return image.reproject({
    crs: 'EPSG:4326',
    scale: 500
  });
});

var forestLAI = reprojectedLAI.map(function(image) {
  return image.updateMask(forestMask).clip(region); 
});

// Calculate the median within the 16-day time intervals
var startDate = ee.Date('2001-01-01'); 
var endDate = ee.Date('2023-12-31'); 

var dateList = ee.List.sequence(0, endDate.difference(startDate, 'day').divide(16).round())
                  .map(function(n) {
                    return startDate.advance(ee.Number(n).multiply(16), 'day');
                  });

var resampledLAI = ee.ImageCollection.fromImages(
  dateList.map(function(date) {
    date = ee.Date(date); 
    var medianImage = forestLAI.filterDate(date, date.advance(16, 'day'))
                               .median(); 
    return medianImage.set('system:time_start', date.millis()); 
  })
);


var laiList = resampledLAI.toList(resampledLAI.size());

var listSize = laiList.size().getInfo(); 

// export images
for (var i = 0; i < listSize; i++) {
  var image = ee.Image(laiList.get(i));
  var date = ee.Date(image.get('system:time_start')).format('yyyyMMdd').getInfo();

  Export.image.toDrive({
    image: image.select(0), 
    description: 'ForestLAI_' + date,
    folder: 'LAI_Export',
    fileNamePrefix: 'ForestLAI_' + date,
    region: region,
    scale: 500,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}

