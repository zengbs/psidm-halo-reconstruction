typedef double cmpx[2];
typedef float cmpxf[2];

extern int tpB_x, tpB_y, tpB_z;
extern int bpG_x, bpG_y, bpG_z;
extern int Lidx_global;
extern int    *rfunc_num;
extern char *filename_r, *filename_i;
extern double **rv;
extern double ***rfunc;
extern cmpx ***amplitude;

__device__ void cal_index_r(int rsiz, float r, float dr, int *index_r, float *frac_r);
__global__ void cal_and_expand_ylm_O(int rsiz, int nx, int ny, int nz, int octant, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, double ***rfunc, cmpx ***amplitude, int l_max, int l_init);
__global__ void cal_and_expand_ylm(int rsiz, int nx, int ny, int nz, float dx, float dy, float dz, float xcen, float ycen, float zcen, float dr, float r_max, float *arr_r, float *arr_i, int *rfunc_num, double ***rfunc, cmpx ***amplitude, int l_max, int l_init);

void ylm_init(int l_init,int l_max);
void ylm_fin(int l_init,int l_max);
void free_egn(int l_max,int l_init);
//void load_eigenstates(int *l_init,int *l_max, int *ln_total_count, char *filename,int lnodidx);
void load_eigenstates(int *l_init,int *l_max, char *filename,int lnodidx);
void store_ylm(int l_init, int l_max, char *filename);
void load_ylm(int l_init,int l_max, char *filename,int lnodidx);
//void show_ylm(int l_init,int l_max, char *filename,int iter);
void store_eigenstates(int l_max, char *filename);
//void decompose_ylm(double r, double **ylm, double arr_r, double arr_i, int l_max);
void do_expand_ylm(float *arr_r, float *arr_i, float *cal_and_expand_timer, int l_max,int l_init, bool timing_flag);
//void gen_amp(int l_init,int l_max,double A,double beta,double Ec,double eta0,double amp_gs_r,double amp_gs_i,bool israndom);
void read_amp(int l_max,char *filename);

// tunable parameters
