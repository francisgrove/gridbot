uint8 STATUS_CORRECT=0          # received == expected marker
uint8 STATUS_WRONG_FIELD=1      # received != expected marker
uint8 STATUS_LOST=2             # didn't finish the route (timed out trying to find it)
uint8 STATUS_COMPLETED=3        # finished route correctly

int32 run_id
int32 step_number
int32 total_steps
uint8 status

SimpleAruco target_marker
SimpleAruco actual_marker

builtin_interfaces/Duration step_duration
float64 mean_deviation_px
float64 min_deviation_px
float64 max_deviation_px
