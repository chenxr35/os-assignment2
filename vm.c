#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void){
  int addresses[1000];//声明存储逻辑地址的数组
  FILE *fid;//输入文件指针
  FILE *fptr;//输出文件指针
  fid=fopen("addresses.txt","r");//打开存储逻辑地址的文件
  fptr=fopen("out.txt","w+");//打开输出文件
  int i=0;
  while(fscanf(fid,"%d",&addresses[i])!=EOF){
    i++;
  }//读取逻辑地址
  fclose(fid);//关闭文件
  int table[256];//页表
  char memory[256][256];//物理内存
  int frame_index=0;//帧索引，方便物理内存的访问
  int TLB[16][2];//TLB[i][0]存储页码，TLB[i][1]存储对应帧码
  int TLB_index=0;//TLB索引
  for(i=0;i<16;i++){
    for(int j=0;j<2;j++)
      TLB[i][j]=-1;//将TLB设定为未存取状态
  }
  for(int j=0;j<256;j++){
    table[j]=-1;//将页表设定为未存取状态
  }
  int temp;
  int binary[1000][16];//将逻辑地址转换为二进制储存在二维数组binary中
  for(i=0;i<1000;i++){
    temp=addresses[i];
    int j=0;
    while(j<16){
      if(temp>=1){
        binary[i][j]=temp%2;
        temp=temp/2;
      }
      else
        binary[i][j]=0;
      j++;
    }
  }
  for(i=0;i<1000;i++){
    int offset=0;
    for(int j=0;j<8;j++)
      offset=offset+binary[i][j]*pow(2,j);//由逻辑地址计算获取偏移
    int page=0;
    for(int j=8;j<16;j++)
      page=page+binary[i][j]*pow(2,j-8);//由逻辑地址计算获取页码
    //由页码查找获取帧码
    int frame;
    //在TLB中查找
    int flag=0;//TLB是否查找成功的标记值
    for(int j=0;j<16;j++){
      if(TLB[j][0]==page){
        frame=TLB[j][1];
        flag=1;//查找成功
      }
    }
    if(flag!=1){//TLB查找失败，开始查找页表
      if(table[page]==-1){
        FILE *file;//文件指针
        char *buffer;//缓冲区
        file=fopen("BACKING_STORE.bin","rb");//打开后备存储文件
        fseek(file,256*page,0);//定位到页码对应位置
        buffer=(char*)malloc(257);//分配内存
        fread(buffer,1,256,file);//读入数据
        for(int j=0;j<256;j++)
          memory[frame_index][j]=buffer[j];//存入物理内存中帧码对应位置
        free(buffer);//释放内存
        fclose(file);//关闭文件
        table[page]=frame_index;//给页表中页码分配帧码
        //将页码和帧码存入TLB
        if(TLB_index<16){
          TLB[TLB_index][0]=page;//存入页码
          TLB[TLB_index][1]=frame_index;//存入帧码
          TLB_index++;
        }
        frame=frame_index;
        frame_index++;
      }
      else//页表查找成功
        frame=table[page];
    }	
    int phyaddr;
    phyaddr=frame*256+offset;//由帧码和偏移计算物理地址
    //将逻辑地址、物理地址以及物理地址处带符号字节的值
    char output[100]="Virtual address: ";
    char VirtualAddress[10];
    sprintf(VirtualAddress,"%d",addresses[i]);
    strcat(output,VirtualAddress);
    strcat(output," Physical address: ");
    char PhysicalAddress[10];
    sprintf(PhysicalAddress,"%d",phyaddr);
    strcat(output,PhysicalAddress);
    strcat(output," Value: ");
    char Value[10];
    sprintf(Value,"%d",memory[frame][offset]);
    strcat(output,Value);
    strcat(output,"\n");
    fputs(output,fptr);
  }
  fclose(fptr);//关闭文件
  return 0;  
}
