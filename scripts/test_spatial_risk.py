from backend.services.spatial_risk_service import (
    get_india_risk_summary,
    get_state_risk_summary,
    get_district_context,
    compute_spatial_grid_risk,
)

print('--- Testing India View ---')
india = get_india_risk_summary()
print('Total states:', india['total_states'], 'features:', len(india['geojson']['features']))

print('--- Testing State View (Bihar) ---')
state = get_state_risk_summary('Bihar')
print('State:', state['state'], 'districts:', state['total_districts'])

print('--- Testing District View (Patna) ---')
dist = get_district_context('Bihar', 'Patna')
print('District:', dist['district'], f'center: ({dist["lat"]}, {dist["lon"]})')

print('--- Testing Local Grid View (Patna) ---')
grid = compute_spatial_grid_risk(25.5941, 85.1376, radius_m=2000, grid_size_m=500)
print('Total grid cells:', grid['total_cells'])
print('WL counts:', grid['summary']['waterlogging']['counts'])
print('DR counts:', grid['summary']['drainage']['counts'])
print('Hotspots count:', grid['summary']['hotspots_count'])
print('Timings ms:', grid['timings_ms'])
first_feat = grid['geojson']['features'][0]
print('Sample cell properties keys:', list(first_feat['properties'].keys()))
print('Sample cell ID:', first_feat['properties']['grid_id'])
print('Sample WL prob:', first_feat['properties']['waterlogging_probability'])
print('Sample DR prob:', first_feat['properties']['drainage_probability'])
print('Sample Hotspot status:', first_feat['properties']['combined_hotspot_status'])
print('GeoJSON geometry type:', first_feat['geometry']['type'])
print('GeoJSON polygon vertices count:', len(first_feat['geometry']['coordinates'][0]))
