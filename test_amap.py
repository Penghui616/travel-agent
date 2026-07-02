from utils.amap_tools import weather_tool, attraction_tool, restaurant_tool, hotel_tool, distance_tool

city = "南京"

weather = weather_tool(city)
print("天气：", weather)

attractions = attraction_tool(city, ["夜景", "拍照"])
print("景点：", attractions)

restaurants = restaurant_tool(city)
print("餐厅：", restaurants)

hotels = hotel_tool(city)
print("酒店：", hotels)

distance = distance_tool(attractions["attractions"])
print("距离：", distance)