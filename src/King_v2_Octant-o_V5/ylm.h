/* add timing_flag for do_expand_ylm() 2020.12.31 */
void ylm_init(int l_init,int l_max);
void ylm_fin(int l_init,int l_max);
void store_ylm(int l_init,int l_max, char *filename);
void load_ylm(int l_init,int l_max, char *filename,int lnodidx);
void show_ylm(int l_init,int l_max, char *filename,int iter);
void store_eigenstates(int l_max, char *filename);
void free_egn(int l_max,int l_init);
void load_eigenstates(int *l_init,int *l_max, char *filename,int lnodidx);
void cal_index_r(double r, int *index_r, double *frac_r);
void decompose_ylm(double r, double **ylm, double arr_r, double arr_i, int l_max);
void expand_ylm(double r, double **ylm, double *arr_r, double *arr_i, int l_max,int l_init);
void do_decompose_ylm(double x, double y, double z, double r, double arr_r, double arr_i, int l_max);
// add timing_flag for do_expand_ylm() 2020.12.31
void do_expand_ylm(double x, double y, double z, double r, double *arr_r, double *arr_i, int l_max,int l_init,bool timing_flag);
//
void cal_ylm(double x,double y,double z, int l_max);
void show_pow(int l_max, char *file);
void gen_amp(int l_init,int l_max,double A,double beta,double Ec,double eta0,double amp_gs_r,double amp_gs_i,bool israndom);
void read_amp(int l_max,char *filename);

// tunable parameters
const int real_ylm  = 1;

