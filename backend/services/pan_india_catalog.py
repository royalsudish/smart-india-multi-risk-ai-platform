"""
Pan-India Geographic Catalog
Contains standardized geographic bounds, centroids, and major district metadata
for all 28 States and 8 Union Territories across India.
"""

from typing import Dict, Any, List, Optional

# Standard catalog of all 28 States and 8 Union Territories of India
INDIA_STATES: Dict[str, Dict[str, Any]] = {
    "Andhra Pradesh": {
        "code": "AP",
        "capital": "Amaravati",
        "lat": 15.9129,
        "lon": 79.7400,
        "bbox": [76.76, 12.62, 84.75, 19.92],
        "region": "South",
        "districts": [
            {"name": "Guntur", "lat": 16.3067, "lon": 80.4365},
            {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
            {"name": "Vijayawada (NTR)", "lat": 16.5062, "lon": 80.6480},
            {"name": "Tirupati", "lat": 13.6288, "lon": 79.4192},
            {"name": "Kurnool", "lat": 15.8281, "lon": 78.0373},
            {"name": "Nellore", "lat": 14.4426, "lon": 79.9865},
            {"name": "Kakinada", "lat": 16.9891, "lon": 82.2475},
            {"name": "Anantapur", "lat": 14.6819, "lon": 77.6006},
            {"name": "Kadapa", "lat": 14.4673, "lon": 78.8242},
            {"name": "Srikakulam", "lat": 18.2949, "lon": 83.8938},
        ],
    },
    "Arunachal Pradesh": {
        "code": "AR",
        "capital": "Itanagar",
        "lat": 27.1004,
        "lon": 93.6166,
        "bbox": [91.50, 26.47, 97.40, 29.50],
        "region": "North-East",
        "districts": [
            {"name": "Papum Pare (Itanagar)", "lat": 27.1004, "lon": 93.6166},
            {"name": "Tawang", "lat": 27.5861, "lon": 91.8653},
            {"name": "West Kameng", "lat": 27.2645, "lon": 92.4159},
            {"name": "East Siang (Pasighat)", "lat": 28.0667, "lon": 95.3333},
            {"name": "Changlang", "lat": 27.1265, "lon": 95.7381},
        ],
    },
    "Assam": {
        "code": "AS",
        "capital": "Dispur",
        "lat": 26.2006,
        "lon": 92.9376,
        "bbox": [89.70, 24.15, 96.02, 27.97],
        "region": "North-East",
        "districts": [
            {"name": "Guwahati (Kamrup Metro)", "lat": 26.1445, "lon": 91.7362},
            {"name": "Dibrugarh", "lat": 27.4728, "lon": 94.9120},
            {"name": "Silchar (Cachar)", "lat": 24.8333, "lon": 92.7789},
            {"name": "Jorhat", "lat": 26.7509, "lon": 94.2037},
            {"name": "Tezpur (Sonitpur)", "lat": 26.6528, "lon": 92.7926},
            {"name": "Nagaon", "lat": 26.3465, "lon": 92.6840},
            {"name": "Dhubri", "lat": 26.0208, "lon": 89.9753},
            {"name": "Barpeta", "lat": 26.3216, "lon": 91.0057},
        ],
    },
    "Bihar": {
        "code": "BR",
        "capital": "Patna",
        "lat": 25.0961,
        "lon": 85.3131,
        "bbox": [83.33, 24.28, 88.29, 27.52],
        "region": "East",
        "districts": [
            {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
            {"name": "Gaya", "lat": 24.7914, "lon": 85.0002},
            {"name": "Muzaffarpur", "lat": 26.1209, "lon": 85.3647},
            {"name": "Bhagalpur", "lat": 25.2425, "lon": 86.9842},
            {"name": "Darbhanga", "lat": 26.1542, "lon": 85.8918},
            {"name": "Purnia", "lat": 25.7771, "lon": 87.4753},
            {"name": "Begusarai", "lat": 25.4182, "lon": 86.1272},
            {"name": "Nalanda (Bihar Sharif)", "lat": 25.1982, "lon": 85.5149},
            {"name": "Rohtas (Sasaram)", "lat": 24.9528, "lon": 84.0315},
            {"name": "Katihar", "lat": 25.5541, "lon": 87.5719},
        ],
    },
    "Chhattisgarh": {
        "code": "CG",
        "capital": "Raipur",
        "lat": 21.2787,
        "lon": 81.8661,
        "bbox": [80.25, 17.78, 84.40, 24.10],
        "region": "Central",
        "districts": [
            {"name": "Raipur", "lat": 21.2514, "lon": 81.6296},
            {"name": "Bhilai (Durg)", "lat": 21.1904, "lon": 81.2849},
            {"name": "Bilaspur", "lat": 22.0797, "lon": 82.1409},
            {"name": "Korba", "lat": 22.3595, "lon": 82.7501},
            {"name": "Jagdalpur (Bastar)", "lat": 19.0744, "lon": 82.0081},
            {"name": "Rajnandgaon", "lat": 21.0971, "lon": 81.0366},
            {"name": "Ambikapur (Surguja)", "lat": 23.1186, "lon": 83.1979},
        ],
    },
    "Goa": {
        "code": "GA",
        "capital": "Panaji",
        "lat": 15.2993,
        "lon": 74.1240,
        "bbox": [73.67, 14.89, 74.34, 15.80],
        "region": "West",
        "districts": [
            {"name": "North Goa (Panaji)", "lat": 15.4909, "lon": 73.8278},
            {"name": "South Goa (Margao)", "lat": 15.2832, "lon": 73.9862},
            {"name": "Vasco da Gama", "lat": 15.3982, "lon": 73.8113},
            {"name": "Mapusa", "lat": 15.5937, "lon": 73.8142},
        ],
    },
    "Gujarat": {
        "code": "GJ",
        "capital": "Gandhinagar",
        "lat": 22.2587,
        "lon": 71.1924,
        "bbox": [68.12, 20.10, 74.47, 24.71],
        "region": "West",
        "districts": [
            {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
            {"name": "Surat", "lat": 21.1702, "lon": 72.8311},
            {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812},
            {"name": "Rajkot", "lat": 22.3039, "lon": 70.8022},
            {"name": "Bhavnagar", "lat": 21.7645, "lon": 72.1519},
            {"name": "Jamnagar", "lat": 22.4707, "lon": 70.0577},
            {"name": "Gandhinagar", "lat": 23.2156, "lon": 72.6369},
            {"name": "Kutch (Bhuj)", "lat": 23.2420, "lon": 69.6669},
            {"name": "Junagadh", "lat": 21.5222, "lon": 70.4579},
        ],
    },
    "Haryana": {
        "code": "HR",
        "capital": "Chandigarh",
        "lat": 29.0588,
        "lon": 76.0856,
        "bbox": [74.45, 27.65, 77.60, 30.92],
        "region": "North",
        "districts": [
            {"name": "Gurugram", "lat": 28.4595, "lon": 77.0266},
            {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178},
            {"name": "Panipat", "lat": 29.3909, "lon": 76.9635},
            {"name": "Ambala", "lat": 30.3782, "lon": 76.7767},
            {"name": "Hisar", "lat": 29.1492, "lon": 75.7217},
            {"name": "Karnal", "lat": 29.6857, "lon": 76.9905},
            {"name": "Rohtak", "lat": 28.8955, "lon": 76.6066},
            {"name": "Panchkula", "lat": 30.6942, "lon": 76.8606},
        ],
    },
    "Himachal Pradesh": {
        "code": "HP",
        "capital": "Shimla",
        "lat": 31.1048,
        "lon": 77.1734,
        "bbox": [75.60, 30.38, 79.00, 33.22],
        "region": "North",
        "districts": [
            {"name": "Shimla", "lat": 31.1048, "lon": 77.1734},
            {"name": "Dharamshala (Kangra)", "lat": 32.2190, "lon": 76.3234},
            {"name": "Mandi", "lat": 31.7087, "lon": 76.9320},
            {"name": "Kullu", "lat": 31.9579, "lon": 77.1095},
            {"name": "Solan", "lat": 30.9045, "lon": 77.0967},
            {"name": "Chamba", "lat": 32.5534, "lon": 76.1258},
            {"name": "Hamirpur", "lat": 31.6862, "lon": 76.5213},
        ],
    },
    "Jharkhand": {
        "code": "JH",
        "capital": "Ranchi",
        "lat": 23.6102,
        "lon": 85.2799,
        "bbox": [83.32, 21.97, 87.95, 25.35],
        "region": "East",
        "districts": [
            {"name": "Ranchi", "lat": 23.3441, "lon": 85.3096},
            {"name": "Jamshedpur (East Singhbhum)", "lat": 22.8046, "lon": 86.2029},
            {"name": "Dhanbad", "lat": 23.7957, "lon": 86.4304},
            {"name": "Bokaro", "lat": 23.6693, "lon": 86.1511},
            {"name": "Deoghar", "lat": 24.4826, "lon": 86.7001},
            {"name": "Hazaribagh", "lat": 23.9964, "lon": 85.3691},
            {"name": "Giridih", "lat": 24.1852, "lon": 86.3094},
        ],
    },
    "Karnataka": {
        "code": "KA",
        "capital": "Bengaluru",
        "lat": 15.3173,
        "lon": 75.7139,
        "bbox": [74.05, 11.59, 78.58, 18.45],
        "region": "South",
        "districts": [
            {"name": "Bengaluru Urban", "lat": 12.9716, "lon": 77.5946},
            {"name": "Mysuru", "lat": 12.2958, "lon": 76.6394},
            {"name": "Mangaluru (Dakshina Kannada)", "lat": 12.9141, "lon": 74.8560},
            {"name": "Hubballi-Dharwad", "lat": 15.3647, "lon": 75.1240},
            {"name": "Belagavi", "lat": 15.8497, "lon": 74.4977},
            {"name": "Kalaburagi (Gulbarga)", "lat": 17.3297, "lon": 76.8343},
            {"name": "Shivamogga", "lat": 13.9299, "lon": 75.5681},
            {"name": "Ballari", "lat": 15.1394, "lon": 76.9214},
            {"name": "Udupi", "lat": 13.3409, "lon": 74.7421},
        ],
    },
    "Kerala": {
        "code": "KL",
        "capital": "Thiruvananthapuram",
        "lat": 10.8505,
        "lon": 76.2711,
        "bbox": [74.86, 8.30, 77.60, 12.80],
        "region": "South",
        "districts": [
            {"name": "Kochi (Ernakulam)", "lat": 9.9312, "lon": 76.2673},
            {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366},
            {"name": "Kozhikode", "lat": 11.2588, "lon": 75.7804},
            {"name": "Thrissur", "lat": 10.5276, "lon": 76.2144},
            {"name": "Kollam", "lat": 8.8932, "lon": 76.6141},
            {"name": "Alappuzha", "lat": 9.4981, "lon": 76.3388},
            {"name": "Palakkad", "lat": 10.7867, "lon": 76.6548},
            {"name": "Kannur", "lat": 11.8745, "lon": 75.3704},
            {"name": "Wayanad (Kalpetta)", "lat": 11.6050, "lon": 76.0828},
            {"name": "Idukki (Painavu)", "lat": 9.8494, "lon": 76.9804},
        ],
    },
    "Madhya Pradesh": {
        "code": "MP",
        "capital": "Bhopal",
        "lat": 22.9734,
        "lon": 78.6569,
        "bbox": [74.03, 21.08, 82.81, 26.87],
        "region": "Central",
        "districts": [
            {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
            {"name": "Indore", "lat": 22.7196, "lon": 75.8577},
            {"name": "Jabalpur", "lat": 23.1815, "lon": 79.9864},
            {"name": "Gwalior", "lat": 26.2183, "lon": 78.1828},
            {"name": "Ujjain", "lat": 23.1765, "lon": 75.7885},
            {"name": "Sagar", "lat": 23.8388, "lon": 78.7378},
            {"name": "Rewa", "lat": 24.5362, "lon": 81.3037},
            {"name": "Satna", "lat": 24.6005, "lon": 80.8322},
        ],
    },
    "Maharashtra": {
        "code": "MH",
        "capital": "Mumbai",
        "lat": 19.7515,
        "lon": 75.7139,
        "bbox": [72.63, 15.60, 80.90, 22.03],
        "region": "West",
        "districts": [
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
            {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
            {"name": "Thane", "lat": 19.2183, "lon": 72.9781},
            {"name": "Nashik", "lat": 19.9975, "lon": 73.7898},
            {"name": "Chhatrapati Sambhajinagar (Aurangabad)", "lat": 19.8762, "lon": 75.3433},
            {"name": "Solapur", "lat": 17.6599, "lon": 75.9064},
            {"name": "Kolhapur", "lat": 16.7050, "lon": 74.2433},
            {"name": "Amravati", "lat": 20.9374, "lon": 77.7796},
            {"name": "Navi Mumbai", "lat": 19.0330, "lon": 73.0297},
        ],
    },
    "Manipur": {
        "code": "MN",
        "capital": "Imphal",
        "lat": 24.6637,
        "lon": 93.9063,
        "bbox": [93.03, 23.83, 94.78, 25.68],
        "region": "North-East",
        "districts": [
            {"name": "Imphal West", "lat": 24.8170, "lon": 93.9368},
            {"name": "Imphal East", "lat": 24.7964, "lon": 93.9534},
            {"name": "Churachandpur", "lat": 24.3333, "lon": 93.6667},
            {"name": "Thoubal", "lat": 24.6333, "lon": 94.0000},
            {"name": "Bishnupur", "lat": 24.6324, "lon": 93.7634},
        ],
    },
    "Meghalaya": {
        "code": "ML",
        "capital": "Shillong",
        "lat": 25.4670,
        "lon": 91.3662,
        "bbox": [89.81, 25.02, 92.80, 26.07],
        "region": "North-East",
        "districts": [
            {"name": "East Khasi Hills (Shillong)", "lat": 25.5788, "lon": 91.8933},
            {"name": "West Garo Hills (Tura)", "lat": 25.5141, "lon": 90.2033},
            {"name": "Ri Bhoi (Nongpoh)", "lat": 25.9000, "lon": 91.8833},
            {"name": "West Jaintia Hills (Jowai)", "lat": 25.4500, "lon": 92.2000},
            {"name": "Sohra (Cherrapunji)", "lat": 25.2700, "lon": 91.7300},
        ],
    },
    "Mizoram": {
        "code": "MZ",
        "capital": "Aizawl",
        "lat": 23.1645,
        "lon": 92.9376,
        "bbox": [92.25, 21.96, 93.44, 24.52],
        "region": "North-East",
        "districts": [
            {"name": "Aizawl", "lat": 23.7271, "lon": 92.7176},
            {"name": "Lunglei", "lat": 22.8671, "lon": 92.7656},
            {"name": "Champhai", "lat": 23.4750, "lon": 93.3283},
            {"name": "Kolasib", "lat": 24.2241, "lon": 92.6781},
            {"name": "Serchhip", "lat": 23.3100, "lon": 92.8300},
        ],
    },
    "Nagaland": {
        "code": "NL",
        "capital": "Kohima",
        "lat": 26.1584,
        "lon": 94.5624,
        "bbox": [93.33, 25.10, 95.25, 27.04],
        "region": "North-East",
        "districts": [
            {"name": "Kohima", "lat": 25.6751, "lon": 94.1086},
            {"name": "Dimapur", "lat": 25.9090, "lon": 93.7266},
            {"name": "Mokokchung", "lat": 26.3248, "lon": 94.5160},
            {"name": "Tuensang", "lat": 26.2800, "lon": 94.8300},
            {"name": "Wokha", "lat": 26.1000, "lon": 94.2600},
        ],
    },
    "Odisha": {
        "code": "OD",
        "capital": "Bhubaneswar",
        "lat": 20.9517,
        "lon": 85.0985,
        "bbox": [81.37, 17.82, 87.50, 22.57],
        "region": "East",
        "districts": [
            {"name": "Bhubaneswar (Khurda)", "lat": 20.2961, "lon": 85.8245},
            {"name": "Cuttack", "lat": 20.4625, "lon": 85.8830},
            {"name": "Rourkela (Sundargarh)", "lat": 22.2604, "lon": 84.8536},
            {"name": "Puri", "lat": 19.8135, "lon": 85.8312},
            {"name": "Berhampur (Ganjam)", "lat": 19.3150, "lon": 84.7941},
            {"name": "Sambalpur", "lat": 21.4669, "lon": 83.9812},
            {"name": "Balasore", "lat": 21.4934, "lon": 86.9135},
            {"name": "Bhadrak", "lat": 21.0574, "lon": 86.5160},
        ],
    },
    "Punjab": {
        "code": "PB",
        "capital": "Chandigarh",
        "lat": 31.1471,
        "lon": 75.3412,
        "bbox": [73.88, 29.53, 76.92, 32.50],
        "region": "North",
        "districts": [
            {"name": "Ludhiana", "lat": 30.9010, "lon": 75.8573},
            {"name": "Amritsar", "lat": 31.6340, "lon": 74.8723},
            {"name": "Jalandhar", "lat": 31.3260, "lon": 75.5762},
            {"name": "Patiala", "lat": 30.3398, "lon": 76.3869},
            {"name": "Bathinda", "lat": 30.2110, "lon": 74.9455},
            {"name": "Mohali (SAS Nagar)", "lat": 30.7046, "lon": 76.7179},
            {"name": "Pathankot", "lat": 32.2686, "lon": 75.6499},
            {"name": "Hoshiarpur", "lat": 31.5273, "lon": 75.9149},
        ],
    },
    "Rajasthan": {
        "code": "RJ",
        "capital": "Jaipur",
        "lat": 27.0238,
        "lon": 74.2179,
        "bbox": [69.48, 23.05, 78.28, 30.20],
        "region": "North",
        "districts": [
            {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
            {"name": "Jodhpur", "lat": 26.2389, "lon": 73.0243},
            {"name": "Udaipur", "lat": 24.5854, "lon": 73.7125},
            {"name": "Kota", "lat": 25.2138, "lon": 75.8648},
            {"name": "Bikaner", "lat": 28.0229, "lon": 73.3119},
            {"name": "Ajmer", "lat": 26.4499, "lon": 74.6399},
            {"name": "Alwar", "lat": 27.5530, "lon": 76.6346},
            {"name": "Bhilwara", "lat": 25.3407, "lon": 74.6313},
            {"name": "Jaisalmer", "lat": 26.9157, "lon": 70.9083},
        ],
    },
    "Sikkim": {
        "code": "SK",
        "capital": "Gangtok",
        "lat": 27.5330,
        "lon": 88.5122,
        "bbox": [88.00, 27.08, 88.92, 28.12],
        "region": "North-East",
        "districts": [
            {"name": "East Sikkim (Gangtok)", "lat": 27.3389, "lon": 88.6065},
            {"name": "South Sikkim (Namchi)", "lat": 27.1667, "lon": 88.3500},
            {"name": "West Sikkim (Geyzing)", "lat": 27.2833, "lon": 88.2500},
            {"name": "North Sikkim (Mangan)", "lat": 27.5000, "lon": 88.5333},
        ],
    },
    "Tamil Nadu": {
        "code": "TN",
        "capital": "Chennai",
        "lat": 11.1271,
        "lon": 78.6569,
        "bbox": [76.23, 8.08, 80.35, 13.55],
        "region": "South",
        "districts": [
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
            {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
            {"name": "Madurai", "lat": 9.9252, "lon": 78.1198},
            {"name": "Tiruchirappalli", "lat": 10.7905, "lon": 78.7047},
            {"name": "Salem", "lat": 11.6643, "lon": 78.1460},
            {"name": "Tirunelveli", "lat": 8.7139, "lon": 77.7567},
            {"name": "Erode", "lat": 11.3410, "lon": 77.7172},
            {"name": "Vellore", "lat": 12.9165, "lon": 79.1325},
            {"name": "Thanjavur", "lat": 10.7870, "lon": 79.1378},
            {"name": "Kanyakumari", "lat": 8.0883, "lon": 77.5385},
        ],
    },
    "Telangana": {
        "code": "TS",
        "capital": "Hyderabad",
        "lat": 18.1124,
        "lon": 79.0193,
        "bbox": [77.25, 15.83, 81.33, 19.92],
        "region": "South",
        "districts": [
            {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
            {"name": "Warangal", "lat": 17.9689, "lon": 79.5941},
            {"name": "Nizamabad", "lat": 18.6725, "lon": 78.0941},
            {"name": "Karimnagar", "lat": 18.4386, "lon": 79.1288},
            {"name": "Khammam", "lat": 17.2473, "lon": 80.1514},
            {"name": "Ramagundam (Peddapalli)", "lat": 18.7618, "lon": 79.4746},
            {"name": "Mahbubnagar", "lat": 16.7433, "lon": 77.9984},
            {"name": "Nalgonda", "lat": 17.0575, "lon": 79.2684},
        ],
    },
    "Tripura": {
        "code": "TR",
        "capital": "Agartala",
        "lat": 23.9408,
        "lon": 91.9882,
        "bbox": [91.15, 22.93, 92.34, 24.53],
        "region": "North-East",
        "districts": [
            {"name": "West Tripura (Agartala)", "lat": 23.8315, "lon": 91.2868},
            {"name": "Gomati (Udaipur)", "lat": 23.5333, "lon": 91.4833},
            {"name": "South Tripura (Belonia)", "lat": 23.2500, "lon": 91.4500},
            {"name": "North Tripura (Dharmanagar)", "lat": 24.3833, "lon": 92.1667},
            {"name": "Dhalai (Ambassa)", "lat": 23.9167, "lon": 91.8500},
        ],
    },
    "Uttar Pradesh": {
        "code": "UP",
        "capital": "Lucknow",
        "lat": 26.8467,
        "lon": 80.9462,
        "bbox": [77.08, 23.87, 84.63, 30.40],
        "region": "North",
        "districts": [
            {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
            {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319},
            {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739},
            {"name": "Prayagraj (Allahabad)", "lat": 25.4358, "lon": 81.8463},
            {"name": "Agra", "lat": 27.1767, "lon": 78.0081},
            {"name": "Noida (Gautam Buddha Nagar)", "lat": 28.5355, "lon": 77.3910},
            {"name": "Ghaziabad", "lat": 28.6692, "lon": 77.4538},
            {"name": "Meerut", "lat": 28.9845, "lon": 77.7064},
            {"name": "Gorakhpur", "lat": 26.7606, "lon": 83.3732},
            {"name": "Bareilly", "lat": 28.3670, "lon": 79.4304},
            {"name": "Aligarh", "lat": 27.8974, "lon": 78.0880},
            {"name": "Moradabad", "lat": 28.8354, "lon": 78.7758},
        ],
    },
    "Uttarakhand": {
        "code": "UK",
        "capital": "Dehradun",
        "lat": 30.0668,
        "lon": 79.0193,
        "bbox": [77.58, 28.72, 81.05, 31.45],
        "region": "North",
        "districts": [
            {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322},
            {"name": "Haridwar", "lat": 29.9457, "lon": 78.1642},
            {"name": "Nainital", "lat": 29.3919, "lon": 79.4542},
            {"name": "Rishikesh", "lat": 30.0869, "lon": 78.2676},
            {"name": "Haldwani", "lat": 29.2183, "lon": 79.5130},
            {"name": "Almora", "lat": 29.5971, "lon": 79.6591},
            {"name": "Chamoli (Gopeshwar)", "lat": 30.4100, "lon": 79.3300},
            {"name": "Uttarkashi", "lat": 30.7300, "lon": 78.4500},
            {"name": "Pithoragarh", "lat": 29.5800, "lon": 80.2200},
        ],
    },
    "West Bengal": {
        "code": "WB",
        "capital": "Kolkata",
        "lat": 22.9868,
        "lon": 87.8550,
        "bbox": [85.82, 21.53, 89.88, 27.22],
        "region": "East",
        "districts": [
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
            {"name": "Howrah", "lat": 22.5958, "lon": 88.2636},
            {"name": "Siliguri (Darjeeling)", "lat": 26.7271, "lon": 88.3953},
            {"name": "Asansol (Paschim Bardhaman)", "lat": 23.6739, "lon": 86.9524},
            {"name": "Durgapur", "lat": 23.5204, "lon": 87.3119},
            {"name": "North 24 Parganas (Barasat)", "lat": 22.7230, "lon": 88.4800},
            {"name": "South 24 Parganas (Alipore)", "lat": 22.5300, "lon": 88.3200},
            {"name": "Malda", "lat": 25.0108, "lon": 88.1411},
            {"name": "Murshidabad (Baharampur)", "lat": 24.0984, "lon": 88.2514},
        ],
    },
    # Union Territories
    "Delhi": {
        "code": "DL",
        "capital": "New Delhi",
        "lat": 28.7041,
        "lon": 77.1025,
        "bbox": [76.84, 28.40, 77.35, 28.88],
        "region": "North",
        "districts": [
            {"name": "Central Delhi", "lat": 28.6500, "lon": 77.2200},
            {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "South Delhi", "lat": 28.5355, "lon": 77.2200},
            {"name": "North Delhi", "lat": 28.7100, "lon": 77.1800},
            {"name": "East Delhi", "lat": 28.6300, "lon": 77.3000},
            {"name": "West Delhi", "lat": 28.6600, "lon": 77.0800},
            {"name": "South West Delhi (Dwarka)", "lat": 28.5921, "lon": 77.0460},
        ],
    },
    "Jammu and Kashmir": {
        "code": "JK",
        "capital": "Srinagar / Jammu",
        "lat": 33.7782,
        "lon": 76.5762,
        "bbox": [73.50, 32.20, 78.50, 35.50],
        "region": "North",
        "districts": [
            {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973},
            {"name": "Jammu", "lat": 32.7266, "lon": 74.8570},
            {"name": "Anantnag", "lat": 33.7311, "lon": 75.1522},
            {"name": "Baramulla", "lat": 34.2000, "lon": 74.3400},
            {"name": "Udhampur", "lat": 32.9250, "lon": 75.1417},
            {"name": "Kathua", "lat": 32.3700, "lon": 75.5200},
        ],
    },
    "Ladakh": {
        "code": "LA",
        "capital": "Leh",
        "lat": 34.1526,
        "lon": 77.5771,
        "bbox": [75.50, 32.50, 80.50, 36.00],
        "region": "North",
        "districts": [
            {"name": "Leh", "lat": 34.1526, "lon": 77.5771},
            {"name": "Kargil", "lat": 34.5539, "lon": 76.1349},
        ],
    },
    "Chandigarh": {
        "code": "CH",
        "capital": "Chandigarh",
        "lat": 30.7333,
        "lon": 76.7794,
        "bbox": [76.70, 30.68, 76.85, 30.79],
        "region": "North",
        "districts": [
            {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
        ],
    },
    "Puducherry": {
        "code": "PY",
        "capital": "Puducherry",
        "lat": 11.9416,
        "lon": 79.8083,
        "bbox": [79.65, 11.75, 79.90, 12.05],
        "region": "South",
        "districts": [
            {"name": "Puducherry", "lat": 11.9416, "lon": 79.8083},
            {"name": "Karaikal", "lat": 10.9254, "lon": 79.8380},
            {"name": "Mahe", "lat": 11.7000, "lon": 75.5333},
            {"name": "Yanam", "lat": 16.7333, "lon": 82.2167},
        ],
    },
    "Andaman and Nicobar Islands": {
        "code": "AN",
        "capital": "Port Blair",
        "lat": 11.7401,
        "lon": 92.6586,
        "bbox": [92.15, 6.75, 94.00, 13.70],
        "region": "Islands",
        "districts": [
            {"name": "South Andaman (Port Blair)", "lat": 11.6234, "lon": 92.7265},
            {"name": "North and Middle Andaman (Mayabunder)", "lat": 12.9234, "lon": 92.8988},
            {"name": "Nicobar (Car Nicobar)", "lat": 9.1550, "lon": 92.8180},
        ],
    },
    "Dadra and Nagar Haveli and Daman and Diu": {
        "code": "DN",
        "capital": "Daman",
        "lat": 20.4283,
        "lon": 72.8397,
        "bbox": [72.80, 20.00, 73.15, 20.50],
        "region": "West",
        "districts": [
            {"name": "Daman", "lat": 20.4283, "lon": 72.8397},
            {"name": "Diu", "lat": 20.7144, "lon": 70.9874},
            {"name": "Dadra and Nagar Haveli (Silvassa)", "lat": 20.2763, "lon": 73.0083},
        ],
    },
    "Lakshadweep": {
        "code": "LD",
        "capital": "Kavaratti",
        "lat": 10.5667,
        "lon": 72.6417,
        "bbox": [71.50, 8.00, 74.00, 12.50],
        "region": "Islands",
        "districts": [
            {"name": "Kavaratti", "lat": 10.5667, "lon": 72.6417},
            {"name": "Agatti", "lat": 10.8500, "lon": 72.1833},
            {"name": "Minicoy", "lat": 8.2833, "lon": 73.0500},
        ],
    },
}


def get_all_states() -> List[Dict[str, Any]]:
    """Returns list of all states with metadata and district counts."""
    result = []
    for state_name, s_data in INDIA_STATES.items():
        result.append({
            "state": state_name,
            "code": s_data["code"],
            "capital": s_data["capital"],
            "lat": s_data["lat"],
            "lon": s_data["lon"],
            "bbox": s_data["bbox"],
            "region": s_data["region"],
            "district_count": len(s_data.get("districts", [])),
        })
    return sorted(result, key=lambda x: x["state"])


def find_state(query: str) -> Optional[Dict[str, Any]]:
    """Finds state by exact name, partial name, or 2-letter state code."""
    if not query:
        return None
    q = query.strip().lower().replace("&", "and")
    
    # Check exact match or code match
    for state_name, s_data in INDIA_STATES.items():
        s_norm = state_name.lower().replace("&", "and")
        if s_norm == q or s_data["code"].lower() == q:
            return {"state": state_name, **s_data}
            
    # Check partial match
    for state_name, s_data in INDIA_STATES.items():
        s_norm = state_name.lower().replace("&", "and")
        if q in s_norm or s_norm in q:
            return {"state": state_name, **s_data}
            
    return None


def get_districts_for_state(state_name: str) -> List[Dict[str, Any]]:
    """Returns all districts for a given state."""
    state_obj = find_state(state_name)
    if not state_obj:
        return []
    return state_obj.get("districts", [])


def find_district(state_name: Optional[str], district_name: str) -> Optional[Dict[str, Any]]:
    """Finds district metadata within a state or across India."""
    d_clean = district_name.strip().lower().replace("&", "and")
    
    if state_name:
        s_obj = find_state(state_name)
        if s_obj:
            for dist in s_obj.get("districts", []):
                dist_norm = dist["name"].lower().replace("&", "and")
                if d_clean in dist_norm or dist_norm in d_clean:
                    return {"state": s_obj["state"], **dist}

    # Search all states
    for s_name, s_data in INDIA_STATES.items():
        for dist in s_data.get("districts", []):
            dist_norm = dist["name"].lower().replace("&", "and")
            if d_clean in dist_norm or dist_norm in d_clean:
                return {"state": s_name, **dist}
                
    return None


def resolve_location(lat: float, lon: float) -> Dict[str, Any]:
    """Resolves latitude and longitude to nearest state and district in catalog."""
    lat_f = float(lat)
    lon_f = float(lon)
    
    # 1. Check bounding boxes
    matched_state = None
    for s_name, s_data in INDIA_STATES.items():
        bbox = s_data["bbox"] # [min_lon, min_lat, max_lon, max_lat]
        if bbox[0] <= lon_f <= bbox[2] and bbox[1] <= lat_f <= bbox[3]:
            matched_state = s_name
            break
            
    if not matched_state:
        # Distance to centroid fallback
        best_dist = float("inf")
        for s_name, s_data in INDIA_STATES.items():
            d = (s_data["lat"] - lat_f) ** 2 + (s_data["lon"] - lon_f) ** 2
            if d < best_dist:
                best_dist = d
                matched_state = s_name
                
    s_obj = INDIA_STATES[matched_state]
    
    # Find nearest district in that state
    best_dist = float("inf")
    nearest_district = None
    for dist in s_obj.get("districts", []):
        d = (dist["lat"] - lat_f) ** 2 + (dist["lon"] - lon_f) ** 2
        if d < best_dist:
            best_dist = d
            nearest_district = dist["name"]
            
    return {
        "state": matched_state,
        "state_code": s_obj["code"],
        "district": nearest_district or s_obj["capital"],
        "capital": s_obj["capital"],
        "region": s_obj["region"],
    }
