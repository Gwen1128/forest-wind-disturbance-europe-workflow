// 定义研究范围（欧洲区域）
var region = ee.Geometry.Polygon([
  [[-31.5, 71.2], [41.5, 71.2], [41.5, 34.5], [-31.5, 34.5], [-31.5, 71.2]]
]);

// 加载 Terra 数据 (MOD15A2H)
var terraLAI = ee.ImageCollection("MODIS/061/MOD15A2H")
                .filterDate('2000-02-18', '2023-12-31') // Terra 起始时间
                .filterBounds(region)
                .select('Lai_500m')
                .map(function(image) {
                  return image.multiply(0.1) // 缩放 LAI 值
                              .copyProperties(image, ['system:time_start']);
                });

// 加载 Aqua 数据 (MYD15A2H)
var aquaLAI = ee.ImageCollection("MODIS/061/MYD15A2H")
                .filterDate('2002-07-04', '2024-12-31') // Aqua 起始时间
                .filterBounds(region)
                .select('Lai_500m')
                .map(function(image) {
                  return image.multiply(0.1) // 缩放 LAI 值
                              .copyProperties(image, ['system:time_start']);
                });

// 合并 Aqua 和 Terra 数据
var combinedLAI = aquaLAI.merge(terraLAI)
                         .sort('system:time_start')
                         .map(function(image) {
                           return image.updateMask(image.gte(0).and(image.lte(10))); // 剔除异常值
                         });

// 创建森林掩膜 (Hansen Global Forest Change 2023)
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var forestMask = gfc.select('treecover2000').gte(25); // 树冠覆盖率 >= 25% 视为森林

// 重投影到 EPSG:4326 并裁剪
var reprojectedLAI = combinedLAI.map(function(image) {
  return image.reproject({
    crs: 'EPSG:4326',
    scale: 500
  });
});

var forestLAI = reprojectedLAI.map(function(image) {
  return image.updateMask(forestMask).clip(region); // 应用掩膜并裁剪
});

// 16天时间分辨率中值重采样
var startDate = ee.Date('2001-01-01'); 
var endDate = ee.Date('2023-12-31'); // 数据结束时间

// 生成时间序列中16天的时间间隔
var dateList = ee.List.sequence(0, endDate.difference(startDate, 'day').divide(16).round())
                  .map(function(n) {
                    return startDate.advance(ee.Number(n).multiply(16), 'day');
                  });

// 在16天时间间隔内计算中值
var resampledLAI = ee.ImageCollection.fromImages(
  dateList.map(function(date) {
    date = ee.Date(date); // 确保日期是 ee.Date 类型
    var medianImage = forestLAI.filterDate(date, date.advance(16, 'day'))
                               .median(); // 计算16天内的中值
    return medianImage.set('system:time_start', date.millis()); // 返回影像并设置时间戳
  })
);




// 将影像集合转换为列表
var laiList = resampledLAI.toList(resampledLAI.size());

// 获取列表大小
var listSize = laiList.size().getInfo(); // 将列表大小拉取到客户端

// 遍历列表并逐个导出影像
for (var i = 0; i < listSize; i++) {
  var image = ee.Image(laiList.get(i));
  var date = ee.Date(image.get('system:time_start')).format('yyyyMMdd').getInfo();

  Export.image.toDrive({
    image: image.select(0), // 导出第一个波段，即 LAI 数据
    description: 'ForestLAI_' + date,
    folder: 'LAI_Export',
    fileNamePrefix: 'ForestLAI_' + date,
    region: region,
    scale: 500,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}




// 从影像集合中获取第一张影像
var firstImage = resampledLAI.first();

// 检查是否成功获取影像
print('First Image:', firstImage);

// 可视化参数
var visParams = {
  min: 0,
  max: 10,
  palette: ['ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901', '66a000', '529400']
};

// 在地图上添加第一张影像
Map.centerObject(region, 4); // 以研究区域为中心
Map.addLayer(firstImage, visParams, 'First Resampled LAI');
