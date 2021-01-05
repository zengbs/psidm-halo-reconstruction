/* move extern int Lidx_global and extern int octant_global to here 2020.12.29*/
/* add data dump and restart 2021.01.04 */
extern int Lidx_global;
extern int octant_global;

int get_pos(int i, int j, int k);
/* add data dump and restart 2021.01.04 */
void read_array(bool restart, int restart_id, char *filename_r, char *filename_i);
//void read_array(char *filename_r, char *filename_i);
//
/* add data dump and restart 2021.01.04 */
void write_array(bool dump, int* dump_id, char *filename_r, char *filename_i);
//void write_array(char *filename_r, char *filename_i);
//
void find_cen(void);
void cal_dis(int i, int j, int k, double *x, double *y, double *z, double *r);
void cal_dis_O(int i, int j, int k, double *x, double *y, double *z, double *r,int octant);
void cal_r_max(void);
void cal_r_max_O(void);
void array_pro(double *rho);
void array_decompose_ylm(int l_max);
void array_init();
void array_add_ylm(int l_max,int l_init,bool firstadd);
//void array_add_ylm2(int l_max);
void array_sub_ylm(int l_max,int l_init);
void over_density(double *rho, double target);
void write_sliceXY(char *filename_r, char *filename_i);
void write_sliceYZ(char *filename_r, char *filename_i);
void sample_error(int error, char *string);
void add_global_array();
