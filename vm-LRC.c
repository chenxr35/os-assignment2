#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define length 128

int main(void){
  int addresses[1000];//声明存储逻辑地址的数组
  int page_fault=0;//缺页次数
  int TLB_hit=0;//TLB命中次数
  FILE *fid;//输入文件指针
  FILE *fptr;//输出文件指针
  fid=fopen("addresses.txt","r");//打开存储逻辑地址的文件
  int i=0;
  while(fscanf(fid,"%d",&addresses[i])!=EOF){
    i++;
  }//读取逻辑地址
  fclose(fid);//关闭文件
  int table[length][2];//页表
  int table_index=0;//页表索引
  for(i=0;i<length;i++){
    for(int j=0;j<2;j++)
      table[i][j]=-1;//将页表设定为未存取状态
  }
  char memory[256][256];//物理内存
  int frame_index=0;//帧索引，方便物理内存的访问
  int TLB[16][2];//TLB[i][0]存储页码，TLB[i][1]存储对应帧码
  int TLB_index=0;//TLB索引
  for(i=0;i<16;i++){
    for(int j=0;j<2;j++)
      TLB[i][j]=-1;//将TLB设定为未存取状态
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
        flag=1;//TLB查找成功
        TLB_hit++;//TLB命中次数+1
        if(j!=TLB_index-1){//如果访问的页不在队尾，更新队列，将访问的页移至队尾
          int temp1,temp2;
          temp1=TLB[j][0];
          temp2=TLB[j][1];
          for(int k=j;k<TLB_index-1;k++){
            TLB[k][0]=TLB[k+1][0];
            TLB[k][1]=TLB[k+1][1];
          }
          TLB[TLB_index-1][0]=temp1;
          TLB[TLB_index-1][1]=temp2;
        }
        break;
      }
    }
    if(flag!=1){//TLB查找失败，开始查找页表
      int sign=0;//页表是否查找成功的标记值
      for(int j=0;j<length;j++){
        if(table[j][0]==page){
          frame=table[j][1];
          sign=1;//页表查找成功
          if(j!=table_index-1){//如果访问的页不在队尾，更新队列，将访问的页移至队尾
            int temp1,temp2;
            temp1=table[j][0];
            temp2=table[j][1];
            for(int k=j;k<table_index-1;k++){
              table[k][0]=table[k+1][0];
              table[k][1]=table[k+1][1];
            }
            table[table_index-1][0]=temp1;
            table[table_index-1][1]=temp2;
          }
          break;
        }
      }
     
      if(sign!=1){//页表查找失败
        //所有帧码分配完后，重新寻找不在TLB和页表中的帧码
        if(frame_index>=256){
          int judge[128];
          for(int j=0;j<16;j++)
            judge[TLB[j][1]]=1;
          for(int j=0;j<128;j++)
            judge[table[j][1]]=1;
          for(int j=0;j<128;j++){
            if(judge[j]!=1){
              frame_index=j;//找到空闲帧码
              break;
            }
          }
        }
        page_fault++;//缺页次数+1
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
        //将页码和帧码存入TLB
        if(TLB_index<16){//TLB未满
          TLB[TLB_index][0]=page;//存入页码
          TLB[TLB_index][1]=frame_index;//存入帧码
          TLB_index++;
        }
        else{//TLB已满，LRU策略更新TLB
          for(int j=0;j<15;j++){//将最长时间未使用的页码剔除
            TLB[j][0]=TLB[j+1][0];
            TLB[j][1]=TLB[j+1][1];
          }
          TLB[15][0]=page;//存入新页码
          TLB[15][1]=frame_index;//存入新帧码
        }
        //将页码和帧码存入页表
        if(table_index<length){
          table[table_index][0]=page;
          table[table_index][1]=frame_index;
          table_index++;
        }
        else{//页表已满，LRU策略更新页表
          for(int j=0;j<length-1;j++){//将最长时间未使用的页码剔除
            table[j][0]=table[j+1][0];
            table[j][1]=table[j+1][1];
          }
          table[length-1][0]=page;//存入新页码
          table[length-1][1]=frame_index;//存入新帧码
        }
        frame=frame_index;
        frame_index++;
      }
    }
  }
  double page_fault_rate;//缺页率
  page_fault_rate=(double)page_fault/1000;
  double TLB_hit_rate;//TLB命中率
  TLB_hit_rate=(double)TLB_hit/1000;
  printf("缺页率:%lf\nTLB命中率:%lf\n",page_fault_rate,TLB_hit_rate);
  return 0;  
}
