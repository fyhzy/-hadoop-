## hadoop基础知识学习

### 理论知识

#### **1、大数据的含义是什么？**

大数据是指无法在一定时间范围内用常规软件工具进行捕捉、管理和处理的数据集合。

#### **2、大数据有哪些特征？**

数据体量大，数据类型多，处理速度快，价值密度低。

#### **3、阐述大数据和人工智能之间的关系。**

大数据和人工智能是紧密相关的概念，它们之间存在着相互促进、相互依赖的关系。

首先，大数据为人工智能提供了支撑。人工智能算法的训练和优化通常需要大量的数据作为输入。大数据的积累和存储使得人工智能算法能够从中学习、挖掘并识别模式和规律。大数据为人工智能提供了数据源，使得AI系统能够更好地理解、分析和应用数据，从而实现更准确和高效的决策和预测。

另一方面，人工智能也推动了大数据的发展和应用。人工智能技术可以帮助大数据处理和分析，发现数据中的隐藏信息和有价值的见解。通过人工智能技术，可以建立高效的数据挖掘和分析模型，自动发现和提取数据中的特征和模式，从而加速和提升大数据的处理效率和质量。

此外，大数据和人工智能的结合还推动了更广泛的应用领域的发展。通过分析和利用大数据，人工智能可以在各个领域中应用，包括金融、医疗、交通、零售等。例如，在金融领域，人工智能可以分析大量的金融数据并进行预测和风险评估；在医疗领域，人工智能可以利用大量的医疗数据进行诊断和治疗决策。大数据为人工智能提供了支持，人工智能则通过处理大数据带来了更多的商业和社会价值。

综上所述，大数据和人工智能之间相辅相成，紧密联系在一起。它们的关系不仅体现在数据的支撑和应用上，还推动了更广泛领域的发展和创新。

#### **4、阐述大数据开源项目Hadoop、Spark、Hive和HBase各自的作用。**

Hadoop是一个分布式计算框架，主要用于存储和处理大规模数据集。它的核心组件包括Hadoop Distributed File System（HDFS）和MapReduce。HDFS用于将数据分布式存储在多个计算节点上，而MapReduce提供了一种并行处理模型，可以对存储在HDFS上的数据进行分布式计算和数据处理。

Spark是一个快速、通用的大数据处理引擎。相较于Hadoop的MapReduce，Spark提供了更高级别的API和功能，支持内存计算、迭代计算和流式计算，并具有更好的性能和灵活性。Spark可以与Hadoop等其他大数据工具集成，并提供了丰富的库和工具，例如Spark SQL、Spark Streaming和MLlib，用于数据查询、流处理和机器学习。

Hive是一个基于Hadoop的数据仓库工具，它提供了类似于SQL的查询语言（称为HiveQL），使得用户可以方便地进行数据查询和分析。Hive将结构化查询语句转换为MapReduce或Tez任务来执行，允许开发人员和分析师通过SQL接口来处理大规模的数据集，而无需编写复杂的MapReduce程序。

HBase是一个分布式、可伸缩的NoSQL数据库，设计用于存储大规模结构化数据。它基于Hadoop的HDFS进行数据存储，提供了快速的随机读写能力，并具有自动分片和复制功能。HBase适合存储半结构化和非结构化数据，例如日志数据和传感器数据，可以支持高性能的实时查询和数据访问。

#### Google的三驾马车

##### 1.Google FS

- GFS是一个可扩展的分布式文件系统，用于大型的、分布式的、对大量数据进行访问的应用。它运行于廉价的普通硬件上，提供容错功能。

##### 2.MapReduce

- MapReduce是针对分布式并行计算的一套编程模型。

- MapReduce是由Map和reduce组成,来自于Lisp，Map是影射，把指令分发到多个worker上去，Reduce是规约，把Map的worker计算出来的结果合并。
- Google的MapReduce实现使用GFS存储数据。

##### 3.Bigtable

- 就像文件系统需要数据库来存储结构化数据一样，GFS也需要Bigtable来存储结构化数据。
- BigTable 是建立在 GFS ，Scheduler ，Lock Service 和 MapReduce 之上的。每个Table都是一个多维的稀疏图。
- 为了管理巨大的Table，把Table根据行分割，这些分割后的数据统称为：Tablets。每个Tablets大概有 100-200 MB，每个机器存储100个左右的 Tablets。底层的架构是：GFS。由于GFS是一种分布式的文件系统，采用Tablets的机制后，可以获得很好的负载均衡。比如：可以把经常响应的表移动到其他空闲机器上，然后快速重建。

**hadoop对Google公司三篇论文思想的实现**

- GFS（Google File System）介绍了一个分布式文件系统，它可以管理大规模数据集。Hadoop的HDFS（Hadoop Distributed File System）就是基于GFS思想实现的。
- MapReduce是一种编程模型，它可以处理大规模数据集。Hadoop的MapReduce框架就是基于MapReduce思想实现的。MapReduce将输入数据分割成若干个块，然后进行并行处理，最后将结果汇总起来。
- BigTable是一种大规模分布式存储系统，它可以处理海量的结构化数据。Hadoop的HBase就是基于BigTable思想实现的。HBase是一个面向列的数据库，可以提供高效的随机读写和批量处理能力。

综上所述，Hadoop对Google公司三篇论文思想的实现包括基于GFS思想的HDFS分布式文件系统、基于MapReduce思想的MapReduce框架和基于BigTable思想的HBase数据库。

#### hadoop的三种运行模式

- ##### **单机模式（独立模式）（Local或Standalone Mode）**


默认情况下，Hadoop即处于该模式，用于开发和调式。

不对配置文件进行修改。

使用本地文件系统，而不是分布式文件系统。

Hadoop不会启动NameNode、DataNode、JobTracker、TaskTracker等守护进程。用于对MapReduce程序的逻辑进行调试，确保程序的正确。

- ##### **伪分布式模式（Pseudo-Distrubuted Mode）**


模拟一个小规模的集群,在一台主机模拟多主机。

Hadoop启动NameNode、DataNode、JobTracker、TaskTracker这些守护进 程都在同一台机器上运行，是相互独立的Java进程。

在这种模式下，Hadoop使用的是分布式文件系统，各个作业也是由JobTraker服务，来管理的独立进程。在单机模式之上增加了代码调试功能，允许检查内存使用情况，HDFS输入输出，类似于完全分布式模式。

- ##### **完全分布式模式（Fully-Distributed Mode）**


Hadoop守护进程运行在一个集群上。

本地模式（Local/Standalone Mode）：单台服务器，数据存储在Linux本地。生产环境几乎不会采用该模式

伪分布式模式（Pseudo-Distributed Mode）：单台服务器，数据存储在HDFS上。有较少的小型公司采用该模式。

完全分布式模式（Fully-Distributed Mode）：多台服务器组成集群，数据存储在HDFS上，多台服务器工作。在企业中大量使用。

#### Hdfs基础知识

**Hdfs简介**

一个高可靠，高吞吐量的分布式文件系统，通俗的说就是把文件存储到多台服务器上。

**HDFS组成**

- **Client（客户端）**：文件切分、与NameNode交互、获取文件的位置信息；与DataNode交互，读取或写入文件；Client提供了一些命令来访问和部署HDFS等。
- **NameNode（Master）**：管理整个文件的元数据（命名空间信息，块信息）；数据块映射信息；配置副本策略、处理客户端读写请求。

- **DataNode（Slave）**：存储文件；执行数据块的读写操作。

- **Secondary Namenode** （辅助工作者）：辅助NameNode；执行fsimage和edits的定期合作，并推送给NameNode。

#### YARN的架构及其组件功能

- 
  **ResourceManager：**
  全局的资源管理器，整个集群只有一个，负责集群资源的统一管理和调度分配。

  功能：
  处理客户端请求
  启动/监控Application Master
  监控NodeManager
  资源分配与调度

- **NodeManager：**
  整个集群有多个，负责单节点资源管理和使用。

  功能：
  单个节点上的资源管理和任务管理（监视Container的资源使用情况，不会监视任务）
  向ResourceManager汇报
  处理Application Master的命令

- **Task：**
  应用程序的具体执行任务

- **Container：**
  在YARN中资源分配的基本单位，封存了CPU和
  内存资源的一个容器，相当于一个Task运行环境的抽象。
  一个程序所需的Container分为两种：运行于ApplicationManager的和运行各类Task的。前者类似于是向计算机要整个程序所需要的资源，后者是向ResourceManager要资源

  功能：
  对任务运行环境的抽象
  描述一系列信息
  任务运行资源(节点，内存，cpu)
  任务启动命令

- **Application Master：**
  应用程序管理员，主要负责单个应用程序的管理（集群中可能同时有多个应用程序在运行，每个应用程序都有各自的Application Master）

​		功能：
​		为应用程序向ResourseManager申请资源（CPU、内存）
​		将资源分配给所管理的应用程序的Task
​		启动和监视Task的运行状态和运行进度
​		关闭并释放自己的Container

#### hive，mysql，hbase关系与区别

##### **hive与mysql比较**

1.查询语言不同:hive是hql语言，mysql是sql语言

2.数据存储位置不同:hive是把数据存储到hdfs，而mysql数据存储在自己的系统中

3.数据格式:hive数据格式可以用户自定义，mysql有自己的系统定义格式

4.数据更新:hive不支持数据更新，只可以读，不可以写，sql支持数据的读写

5.索引：hive没有索引，因此查询数据的时候是通过mapreduce很暴力的把数据都查询一遍，也造成了hive查询数据速度很慢的原因，而mysql有索引；

6.延迟性:hive没有索引，因此查询数据的时候通过mapreduce很暴力 的把数据都查询一遍，也造成了hive查询数据速度很慢的原因，而mysql有索引；

7.数据规模:hive存储的数据量超级大，而mysql只是存储一些少量的业务数据

8.底层执行原理：hive底层是用的mapreduce，而mysql是excutor执行器；

Hive 具有 SQL 数据库的外表，但应用场景完全不同，Hive 只适合用来做海量离线数 据统计分析，也就是数据仓库。

##### hive和hbase比较：

共同点：都是架构在hadoop之上的，用hadoop作为底层存储

区别：

1.hive是建立在Hadoop之上为了减少MapReduce jobs编写工作的批处理系统，hbase为了支持Hadoop对实时操作的缺陷。

2.全表扫描数据库用hive+hadoop,索引访问，用hbase+hadoop。

3.hive query速度较慢，hbase是非常高效的，比hive高效的多。

4.hive本身不存储和计算数据，完全依赖于hdfs和MapReduce，hive的表纯逻辑。

5.hive借用hadoop的MapReduce来完成一些hive中的命令执行，用MapReduce计算框架。

6.hbase是物理表，不是逻辑表，提供一个超大内存hash表，搜索引擎通过它来存储索引，方便查询操作，是列存储的。

7.hive用hdfs存储文件，hdfs是底层存储，是存放文件的系统，hbase负责组织文件。

##### mysql和hbase比较:

两者属于不同类型数据库。HBASE是按列存储型数据库，MySQL是关系型数据库。

其中，关系型数据库:

表和表、表和字段、数据和数据存在着关系

关系型数据库优点:

1.数据之间有关系，进行数据的增删改查的时候是非常方便的。

2.关系型数据库是有事务操作的，保证数据的完整性和一致性。

关系型数据库缺点：

1.因为数据和数据是有关系的，底层是运行了大量的算法,大量算法会降低系统的效率，会降低性能

2.面对海量数据的增删改查的时候会显的无能为力

3.海量数据对数据进行维护变得非常的无力,因此，关系型数据库适合处理一般量级的数据。

非关系数据库的（redis和MangDB）为了处理海量数据，非关系数据库设计之初就是为了替代关系型数据库的关系。

非关系型数据库优点：

1.海量数据的增删改查是可以的

2.海量数据的维护和处理非常轻松

非关系型数据库缺点：

1.数据和数据没有关系，他们之间就是单独存在的

2.非关系数据库没有关系，没有强大的事务关系，没有保证数据的完整性和安全性,适合处理海量数据，保证效率，不一定安全（统计数据，例如微博数据）

**Hbase的优点**

1.列的可以动态增加，并且列为空就不存储数据,节省存储空间.

2.Hbase自动切分数据，使得数据存储自动具有水平scalability.

3.Hbase可以提供高并发读写操作的支持

**Hbase的缺点：**

1.不能支持条件查询，只支持按照Row key来查询.

2.暂时不能支持Master server的故障切换,当Master宕机后,整个存储系统就会挂掉.

补充:

1.数据类型，Hbase只有简单的字符类型，所有的类型都是交由用户自己处理，它只保存字符串。而关系数据库有丰富的类型和存储方式。

2.数据操作：HBase只有很简单的插入、查询、删除、清空等操作，表和表之间是分离的，没有复杂的表和表之间的关系，而传统数据库通常有各式各样的函数和连接操作。

3.存储模式：HBase是基于列存储的，每个列族都由几个文件保存，不同的列族的文件时分离的。而传统的关系型数据库是基于表格结构和行模式保存的

4.数据维护，HBase的更新操作不应该叫更新，它实际上是插入了新的数据，而传统数据库是替换修改

5.可伸缩性，Hbase这类分布式数据库就是为了这个目的而开发出来的，所以它能够轻松增加或减少硬件的数量，并且对错误的兼容性比较高。而传统数据库通常需要增加中间层才能实现类似的功能。

**Hive、HBase 和 MySQL 是三种不同的数据库技术，各自具有不同的特点和用途。**

Hive 是基于 Hadoop 的数据仓库工具，用于处理大规模的结构化数据。

它使用类似于 SQL 的查询语言（HQL）来进行数据分析和查询。

Hive 基于 MapReduce 或 Apache Tez 运行，并在底层将查询转换为 MapReduce 任务或 Tez 任务。

Hive 适用于批处理和离线分析，对于实时查询性能较低。

HBase 是一个分布式、面向列的 NoSQL 数据库，建立在 Hadoop 上。

HBase 提供了高可靠性、高性能的随机读写能力，适用于海量数据的实时访问。

HBase 使用键值对的数据模型，数据存储在分布式文件系统上，并支持水平扩展。

MySQL 是一种关系型数据库管理系统（RDBMS），常用于传统的事务处理和数据管理。

MySQL 使用结构化查询语言（SQL）进行数据操作，支持事务处理和复杂的关系型查询。

MySQL 适用于小到中等规模的数据处理，具有良好的数据完整性和 ACID（原子性、一致性、隔离性和持久性）特性。

关系和区别：

Hive 和 HBase 都是构建在 Hadoop 生态系统之上的，用于处理大规模数据。Hive 适用于批处理和离线分析，而 HBase 适用于实时访问和随机读写。

MySQL 是一种传统的关系型数据库，用于常规的事务处理和数据管理，与 Hive 和 HBase 相比，它更适合小规模的数据处理和复杂的关系查询。

Hive 和 HBase 可以与其他工具和框架（如Hadoop、Spark）进行集成，以构建更全面的大数据解决方案。

综上所述，Hive、HBase 和 MySQL 是三种不同的数据库技术，用于不同的数据处理场景和需求。选择使用哪种数据库取决于数据量、查询需求、实时性要求等因素。

### MapReduce大作业

####  Student类

public class Student implements WritableComparable<Student> {

  private int id;

  private String name;

  private int grade;

  private int oscore;

  private int dscore;

  private int coscore;

  public Student() {

​    super();

  }

  public Student(int id, String name, int grade, int oscore, int dscore, int coscore) {

​    super();

​    this.id = id;

​    this.name = name;

​    this.grade = grade;

​    this.oscore = oscore;

​    this.dscore = dscore;

​    this.coscore = coscore;

  }

  public String getName() {

​    return name;

  }

  public void setName(String name) {

​    this.name = name;

  }

  ......(get set)

  public double getAvgScore() {

​    return (this.getCoscore()+this.getDscore()+this.getOscore())/3.0;

  }

**//序列化与反序列化**

  @Override

  public void **readFields**(DataInput input) throws IOException {

​    this.id = input.readInt();

​    this.name = input.readUTF();

​    this.grade = input.readInt();

​    this.oscore = input.readInt();

​    this.dscore = input.readInt();

​    this.coscore = input.readInt();

  }

  @Override

  public void **write**(DataOutput output) throws IOException {

​    output.writeInt(this.id);

​    output.writeUTF(this.name);

​    output.writeInt(this.grade);

​    output.writeInt(this.oscore);

​    output.writeInt(this.dscore);

​    output.writeInt(this.coscore);

  }

  @Override

  public String **toString**() {

​    return id+" "+name+" "+grade+" "+String.format("%.2f",this.getAvgScore());

  }

  public int **compareTo**(Student s) {

​    if(this.getAvgScore()>s.getAvgScore()) {

​      return 1;

​    }else if(this.getAvgScore()<s.getAvgScore()) {

​      return -1;

​    }else {

​      return 0;

​    }

  }

}

#### **Mapper **

public class userloginMapper extends Mapper<LongWritable, Text, Student, IntWritable>{

  @Override

  protected void map(LongWritable key, Text value,Context context) throws InterruptedException, IOException {

​    //拿到一行文本内容，转换成String 类型

​    String line = value.toString();

​    //将这行文本切分成单词

​    String[] words=line.split(",");

​    Student s = new Student();

​    s.setId(Integer.parseInt(words[0]));

​    s.setName(words[1]);

​    s.setGrade(Integer.parseInt(words[2]));

​    s.setOscore(Integer.parseInt(words[3]));

​    s.setDscore(Integer.parseInt(words[4]));

​    s.setCoscore(Integer.parseInt(words[5]));

​    context.write(s, new IntWritable(1));

  }

}

#### **Reducer**

public class userloginReducer extends Reducer<Student, IntWritable, Student, NullWritable>{

  @Override

  protected void reduce(Student key, Iterable<IntWritable> values,Context context) throws InterruptedException, IOException {

​    context.write(key, NullWritable.get()); 

  }

}

#### **Driver**

public class userlogin { 

  public static void main(String[] args) throws Exception{

​     Configuration conf = new Configuration();

​     String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

​     Job userloginJob = Job.getInstance(conf,"user logTime");

​     //指定本job所在的jar包

​     userloginJob.setJarByClass(userlogin.class);

​     //设置wordCountJob所用的mapper逻辑类为哪个类

​     userloginJob.setMapperClass(userloginMapper.class);

​     //设置wordCountJob所用的reducer逻辑类为哪个类

​     userloginJob.setReducerClass(userloginReducer.class);

​     //设置map阶段输出的kv数据类型

​    userloginJob.setMapOutputKeyClass(Student.class);

​     userloginJob.setMapOutputValueClass(IntWritable.class);

​     //设置分区

​     userloginJob.setPartitionerClass(Partitioner.class);

​     userloginJob.setNumReduceTasks(2);

​     //指定Comparator的类

​     userloginJob.setSortComparatorClass(Comparator.class);

​     //设置最终输出的kv数据类型

​     userloginJob.setOutputKeyClass(Student.class);

​     userloginJob.setOutputValueClass(NullWritable.class);

​     //设置要处理的文本数据所存放的路径

​     FileInputFormat.setInputPaths(userloginJob,new Path(otherArgs[0]));

​     FileOutputFormat.setOutputPath(userloginJob,new Path(otherArgs[1]));

​     //提交job给hadoop集群

​     userloginJob.waitForCompletion(true);

  }

}

#### Comparator

public class Comparator extends WritableComparator{

  protected Comparator(){

​    super(Student.class,true);

  }

  public int compare(WritableComparable w1, WritableComparable w2) {

​    Student p1 = (Student)w1;

​    Student p2 = (Student)w2;

​    if(p1.compareTo(p2)==0) {

​      return -1*(p1.getId()>p2.getId()?1:-1);

​    }else {

​      return -1*(p1.getAvgScore()>p2.getAvgScore()?1:-1);

​    }

  }

}

#### Partitioner

public class Partitioner extends org.apache.hadoop.mapreduce.Partitioner<Student,IntWritable> {

  public int getPartition(Student key, IntWritable value,int numPartitions) {

​    return key.getGrade()%2;

  }

}

### 命令操作

#### hive

##### 指定精度取整函数: round

语法: round(double a, int d)

返回值: DOUBLE

说明:返回指定精度d的double类型

##### 幂运算函数: pow

语法: pow(double a, double p)

返回值: double

说明:返回a的p次幂

##### 绝对值函数: abs

语法: abs(double a) abs(int a)

返回值: double    int

说明:返回数值a的绝对值

##### 字符串长度函数：length

语法: length(string A)

返回值: int

说明：返回字符串A的长度

##### 字符串反转函数：reverse

语法: reverse(string A)

返回值: string

说明：返回字符串A的反转结果

##### 字符串连接函数-带分隔符：concat_ws

语法: concat_ws(string SEP, string A, string B…)

返回值: string

说明：返回输入字符串连接后的结果，SEP表示各个字符串间的分隔符

##### 字符串截取函数：substr,substring

语法: substr(string A, int start),substring(string A, int start)

返回值: string

说明：返回字符串A从start位置到结尾的字符串

##### 分割字符串函数: split

语法: split(string str, string pat)

返回值: array

说明:按照pat字符串分割str，会返回分割后的字符串数组

**建表**

create table student(id string.name string) row format delimited fields

terminated by '\t';

**加载数据**

load data local impath '/.../student.txt' overwrite into table

student;

select* from student join math where student.id=math.id;

**Hive** **中用于将查询结果写入到本地文件系统的目录**

insert overwrite local directory  'local_directory_path' select ...

#### hbase

**查看namespace**

​	list_namespace

**创建namespace**

​	create_namespace 'test'

​	list_namespace_tables 'test'

**查看所有表**

​	list

**创建表**

​	create 'test:student','id','info','score'

**查看表的基本信息**

​	describe 'test:student'

**插入数据**

​	put 'test:student', '1','info:gender','M'

​	put 'test:student','2','info:name','wang'

​	put 'test:student', '2','score:math','90'

**获取指定行、指定行中的列族、列的信息**

​	get 'test:student','1'

​	get 'test:student','2'

**查询整表数据**

​	scan 'test:student'

**查询指定列簇的数据**

​	scan 'test:student', {COLUMN=>'info'}

**条件过滤**

**ValueFilter：值过滤器，找到符合值条件的键值对**

​	scan 'test:student', FILTER=>"ValueFilter(=,'binary:wang')"

​	scan 'test:student',FILTER=>"ValueFilter(=,'substring:a')"

**扫描特定行**

**SingleColumnValueFilter：在指定的列族和列中进行比较的值过滤器**

​	scan 'test:student',FILTER=>"SingleColumnValueFilter('info','name',=,'binary:wang')"

**添加列族**

​	alter 'test:student', {NAME => 'teacher', METHOD => 'delete'}

**表的启用/禁用**

​	disable 'test:student'

**删除表**

​	disable之后的表才能删除

​	drop 'test:student'

**删除namespace**

​	drop_namespace ‘test’

