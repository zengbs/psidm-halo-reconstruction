typedef double cmpx[2];
typedef float cmpxf[2];

extern int tpB_x, tpB_y, tpB_z;
extern int bpG_x, bpG_y, bpG_z;
extern int Lidx_global;
extern char *filename_r, *filename_i;
extern char buff_r[], buff_i[];
extern float ***amplitude_r, ***amplitude_i;
extern float ***amplitude_r_host, ***amplitude_i_host;
extern float ***amplitude_r_pointer, ***amplitude_i_pointer;
extern cmpx ***amplitude_host;
extern int *rfunc_num;
extern int *rfunc_num_device;
extern float **rv;
extern float ***rfunc;
extern float **rv_pointer;
extern float ***rfunc_l_pointer;
extern float ***rfunc_pointer;
//extern int *rfunc_num_host;
//extern double **rv_host;
//extern double ***rfunc_host;


__device__ void cal_index_r(int rsiz, float r, float dr, int *index_r, float *frac_r);
__global__ void cal_and_expand_ylm_O(int rsiz, int nx, int ny, int nz, int octant, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, float ***rfunc, float ***amplitude_r, float ***amplitude_i, int l_max, int l_init);
__global__ void cal_and_expand_ylm(int rsiz, int nx, int ny, int nz, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, float ***rfunc, float ***amplitude_r, float ***amplitude_i, int l_max, int l_init);

void ylm_init(int l_init,int l_max);
void ylm_fin(int l_init,int l_max);
void free_egn(int l_max,int l_init);
//void load_eigenstates(int *l_init,int *l_max, int *ln_total_count, char *filename,int lnodidx);
void load_eigenstates(int *l_init,int *l_max, int *eigen_num, char *filename,int lnodidx);
void store_ylm(int l_init, int l_max, char *filename);
void load_ylm(int l_init,int l_max, char *filename,int lnodidx);
//void show_ylm(int l_init,int l_max, char *filename,int iter);
void store_eigenstates(int l_max, char *filename);
void do_expand_ylm(float *arr_r, float *arr_i, float *cal_and_expand_timer, int l_max,int l_init, bool timing_flag);
void read_amp(int l_max,char *filename);

// tunable parameters
