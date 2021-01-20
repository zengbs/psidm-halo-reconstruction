extern int octant_global;
extern int Lidx_global;
extern int N_site;
extern float *array_r_host;
extern float *array_i_host;

__device__ int get_pos(int i, int j, int k, int nx, int ny);
__device__ void cal_dis_x(int i, float dx, float xcen, float *x);
__device__ void cal_dis_y(int j, float dy, float ycen, float *y);
__device__ void cal_dis_z(int k, float dz, float zcen, float *z);
__device__ void cal_dis_x_O(int i, int nx, int octant, float dx, float xcen, float *x);
__device__ void cal_dis_y_O(int j, int ny, int octant, float dy, float ycen, float *y);
__device__ void cal_dis_z_O(int k, int nz, int octant, float dz, float zcen, float *z);
__device__ void cal_dis_r(float *x, float *y, float *z, float *r);

void read_array(bool restart, int restart_id, char *filename_r, char *filename_i);
void write_array(bool dump, int* dump_id, char *filename_r, char *filename_i);
void cal_r_max(void);
void cal_r_max_O(void);
void array_init();
void array_fin();
void array_add_ylm(int l_max,int l_init,bool firstadd);
