int32 run_id
string route_name
string[] route_nodes
int32 route_difficulty              # 0=easy, 1=medium, 2=hard

# route_navigator params
float64 linear_vel
float64 angular_vel
float64 k_p
float64 k_i
float64 k_d

# grid_processor params
int32 centerline_node_offset
int32 centerline_step
float64 line_roi_height
float64 line_roi_width
float64 aruco_roi_width
float64 aruco_roi_height
int32 aruco_padding
float64 line_turning_min_angle

# route_generator params
float64 turn_penalty_factor
