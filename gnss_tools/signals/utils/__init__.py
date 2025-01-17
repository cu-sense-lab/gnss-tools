
# /* flip array direction ------------------------------------------------------*/
# void fliparrays(short *in, int n, short *out)
# {
#     int i;
#     short *tmp;
#     tmp=(short *)malloc(n*sizeof(short));
#     if (tmp!=NULL) {
#         memcpy(tmp,in,n);
#         for(i=0;i<n;i++) out[n-i-1]=tmp[i];
#         free(tmp);
#     }
# }
# /* flip array direction ------------------------------------------------------*/
# void fliparrayc(char *in, int n, char *out)
# {
#     int i;
#     char *tmp;
#     tmp=(char *)malloc(n);
#     if (tmp!=NULL) {
#         memcpy(tmp,in,n);
#         for(i=0;i<n;i++) out[n-i-1]=tmp[i];
#         free(tmp);
#     }
# }
# /* octal to binary -----------------------------------------------------------*/
# void oct2bin(const char *oct, int n, int nbit, char *bin, int skiplast,
#              int flip)
# {
#     int i,j,k,skip;
#     const static char octlist[8][3]={{ 1,1,1},{ 1,1,-1},{ 1,-1,1},{ 1,-1,-1},
#                                      {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
#     skip=3*n-nbit;
#     for (i=j=0;i<n;i++) {
#         for (k=0;k<3;k++) {
#             if (!skiplast&&i==0&&k<skip) continue;
#             if (skiplast&&i==n-1&&k>=3-skip) continue;
#             bin[j]=octlist[oct[i]-'0'][k];
#             j++;
#         }
#     }
#     if (flip) fliparrayc(bin,nbit,bin);
# }
# /* hexadecimal to decimal ----------------------------------------------------*/
# int hexc2dec(char hex)
# {
#     if ('0'<=hex&&'9'>=hex) return (hex-'0');
#     if ('A'<=hex&&'F'>=hex) return (hex+10-'A');
#     if ('a'<=hex&&'f'>=hex) return (hex+10-'a');
#     return 0;
# }
# /* hexadecimal to binary -----------------------------------------------------*/
# void hex2bin(const char *hex, int n, int nbit, short *bin, int skiplast,
#              int flip)
# {
#     int i,j,k,skip;
#     const static char hexlist[16][4]=
#         {{ 1, 1,1,1},{ 1, 1,1,-1},{ 1, 1,-1,1},{ 1, 1,-1,-1},
#          { 1,-1,1,1},{ 1,-1,1,-1},{ 1,-1,-1,1},{ 1,-1,-1,-1},
#          {-1, 1,1,1},{-1, 1,1,-1},{-1, 1,-1,1},{-1, 1,-1,-1},
#          {-1,-1,1,1},{-1,-1,1,-1},{-1,-1,-1,1},{-1,-1,-1,-1}};
    
#     skip=4*n-nbit;
#     for (i=j=0;i<n;i++) {
#         for (k=0;k<4;k++) {
#             if (!skiplast&&i==0&&k<skip) continue;
#             if (skiplast&&i==n-1&&k>=4-skip) continue;
#             bin[j]=hexlist[hexc2dec(hex[i])][k];
#             j++;
#         }
#     }
#     if (flip) fliparrays(bin,nbit,bin);
# }



# /* Neuman-Hoffman code (10bit) -----------------------------------------------*/
# static short *gencode_NH10(int *len, double *crate)
# {
#     short *code;

#     if (!(code=(short *)calloc(LEN_NH10,sizeof(short)))) {
#         return NULL;
#     }
#     code[0]=-1; code[1]=-1; code[2]=-1; code[3]=-1; code[4]= 1;
#     code[5]= 1; code[6]=-1; code[7]= 1; code[8]=-1; code[9]= 1;

#     if (len) *len=LEN_NH10;
#     if (crate) *crate=CRATE_NH10;

#     return code;
# }
# /* Neuman-Hoffman code (20bit) -----------------------------------------------*/
# static short *gencode_NH20(int *len, double *crate)
# {
#     short *code;

#     if (!(code=(short *)calloc(LEN_NH20,sizeof(short)))) {
#         return NULL;
#     }
#     code[ 0]=-1; code[ 1]=-1; code[ 2]=-1; code[ 3]=-1; code[ 4]=-1;
#     code[ 5]= 1; code[ 6]=-1; code[ 7]=-1; code[ 8]= 1; code[ 9]= 1;
#     code[10]=-1; code[11]= 1; code[12]=-1; code[13]= 1; code[14]=-1;
#     code[15]=-1; code[16]= 1; code[17]= 1; code[18]= 1; code[19]=-1;

#     if (len) *len=LEN_NH20;
#     if (crate) *crate=CRATE_NH20;

#     return code;
# }
# /* binary offset carrier (BOC) -------------------------------------------------
# * BOC modulation function
# * args   : short  *cide     I   GNSS code (-1 or 1)
# *          int    *len      I/O code length
# *          double *crate    I/O code chip rate (chip/s)
# *          int    m         I   sub-carrier frequency/1.023MHz
# *          int    n         I   chip frequency/1.023MHz
# * return : short*               BOC modulated code
# *-----------------------------------------------------------------------------*/
# static short *boc(short *code, int *len, double *crate, int m, int n)
# {
#     short *boccode;
#     int i,j,N=2*m/n;

#     if (!(boccode=(short *)calloc(N*(*len),sizeof(short)))) {
#         return NULL;
#     }

#     for (i=0;i<(*len);i++) {
#         for (j=0;j<N;j++) {
#             boccode[N*i+j]=code[i];
#         }
#     }
#     /* mix sub carrier */
#     for (i=0;i<N*(*len)/2;i++) {
#         boccode[2*i]=-boccode[2*i];
#     }

#     /* new code length and rate */
#     (*len)*=N;
#     (*crate)*=N;

#     free(code);
#     return boccode;
# }