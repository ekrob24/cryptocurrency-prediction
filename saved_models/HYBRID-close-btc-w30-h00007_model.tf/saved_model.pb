ф<
Ќ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12unknown8Б 8
|
dense_327/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *!
shared_namedense_327/kernel
u
$dense_327/kernel/Read/ReadVariableOpReadVariableOpdense_327/kernel*
_output_shapes

:2 *
dtype0
t
dense_327/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_327/bias
m
"dense_327/bias/Read/ReadVariableOpReadVariableOpdense_327/bias*
_output_shapes
: *
dtype0
|
dense_328/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*!
shared_namedense_328/kernel
u
$dense_328/kernel/Read/ReadVariableOpReadVariableOpdense_328/kernel*
_output_shapes

:2@*
dtype0
t
dense_328/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_328/bias
m
"dense_328/bias/Read/ReadVariableOpReadVariableOpdense_328/bias*
_output_shapes
:@*
dtype0
|
dense_329/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*!
shared_namedense_329/kernel
u
$dense_329/kernel/Read/ReadVariableOpReadVariableOpdense_329/kernel*
_output_shapes

:`*
dtype0
t
dense_329/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_329/bias
m
"dense_329/bias/Read/ReadVariableOpReadVariableOpdense_329/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_68/lstm_cell_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*,
shared_namelstm_68/lstm_cell_68/kernel

/lstm_68/lstm_cell_68/kernel/Read/ReadVariableOpReadVariableOplstm_68/lstm_cell_68/kernel*
_output_shapes
:	Ќ*
dtype0
Ї
%lstm_68/lstm_cell_68/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*6
shared_name'%lstm_68/lstm_cell_68/recurrent_kernel
 
9lstm_68/lstm_cell_68/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_68/lstm_cell_68/recurrent_kernel*
_output_shapes
:	KЌ*
dtype0

lstm_68/lstm_cell_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ**
shared_namelstm_68/lstm_cell_68/bias

-lstm_68/lstm_cell_68/bias/Read/ReadVariableOpReadVariableOplstm_68/lstm_cell_68/bias*
_output_shapes	
:Ќ*
dtype0

gru_59/gru_cell_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namegru_59/gru_cell_59/kernel

-gru_59/gru_cell_59/kernel/Read/ReadVariableOpReadVariableOpgru_59/gru_cell_59/kernel*
_output_shapes
:	*
dtype0
Ѓ
#gru_59/gru_cell_59/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*4
shared_name%#gru_59/gru_cell_59/recurrent_kernel

7gru_59/gru_cell_59/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_59/gru_cell_59/recurrent_kernel*
_output_shapes
:	2*
dtype0

gru_59/gru_cell_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namegru_59/gru_cell_59/bias

+gru_59/gru_cell_59/bias/Read/ReadVariableOpReadVariableOpgru_59/gru_cell_59/bias*
_output_shapes
:	*
dtype0

lstm_69/lstm_cell_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*,
shared_namelstm_69/lstm_cell_69/kernel

/lstm_69/lstm_cell_69/kernel/Read/ReadVariableOpReadVariableOplstm_69/lstm_cell_69/kernel*
_output_shapes
:	KШ*
dtype0
Ї
%lstm_69/lstm_cell_69/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*6
shared_name'%lstm_69/lstm_cell_69/recurrent_kernel
 
9lstm_69/lstm_cell_69/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_69/lstm_cell_69/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0

lstm_69/lstm_cell_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш**
shared_namelstm_69/lstm_cell_69/bias

-lstm_69/lstm_cell_69/bias/Read/ReadVariableOpReadVariableOplstm_69/lstm_cell_69/bias*
_output_shapes	
:Ш*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/dense_327/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *(
shared_nameAdam/dense_327/kernel/m

+Adam/dense_327/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_327/kernel/m*
_output_shapes

:2 *
dtype0

Adam/dense_327/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_327/bias/m
{
)Adam/dense_327/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_327/bias/m*
_output_shapes
: *
dtype0

Adam/dense_328/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*(
shared_nameAdam/dense_328/kernel/m

+Adam/dense_328/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_328/kernel/m*
_output_shapes

:2@*
dtype0

Adam/dense_328/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_328/bias/m
{
)Adam/dense_328/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_328/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_329/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*(
shared_nameAdam/dense_329/kernel/m

+Adam/dense_329/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_329/kernel/m*
_output_shapes

:`*
dtype0

Adam/dense_329/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_329/bias/m
{
)Adam/dense_329/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_329/bias/m*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_68/lstm_cell_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*3
shared_name$"Adam/lstm_68/lstm_cell_68/kernel/m

6Adam/lstm_68/lstm_cell_68/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_68/lstm_cell_68/kernel/m*
_output_shapes
:	Ќ*
dtype0
Е
,Adam/lstm_68/lstm_cell_68/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*=
shared_name.,Adam/lstm_68/lstm_cell_68/recurrent_kernel/m
Ў
@Adam/lstm_68/lstm_cell_68/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_68/lstm_cell_68/recurrent_kernel/m*
_output_shapes
:	KЌ*
dtype0

 Adam/lstm_68/lstm_cell_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*1
shared_name" Adam/lstm_68/lstm_cell_68/bias/m

4Adam/lstm_68/lstm_cell_68/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_68/lstm_cell_68/bias/m*
_output_shapes	
:Ќ*
dtype0

 Adam/gru_59/gru_cell_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/gru_59/gru_cell_59/kernel/m

4Adam/gru_59/gru_cell_59/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_59/gru_cell_59/kernel/m*
_output_shapes
:	*
dtype0
Б
*Adam/gru_59/gru_cell_59/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*;
shared_name,*Adam/gru_59/gru_cell_59/recurrent_kernel/m
Њ
>Adam/gru_59/gru_cell_59/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_59/gru_cell_59/recurrent_kernel/m*
_output_shapes
:	2*
dtype0

Adam/gru_59/gru_cell_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/gru_59/gru_cell_59/bias/m

2Adam/gru_59/gru_cell_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_59/gru_cell_59/bias/m*
_output_shapes
:	*
dtype0
Ё
"Adam/lstm_69/lstm_cell_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*3
shared_name$"Adam/lstm_69/lstm_cell_69/kernel/m

6Adam/lstm_69/lstm_cell_69/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_69/lstm_cell_69/kernel/m*
_output_shapes
:	KШ*
dtype0
Е
,Adam/lstm_69/lstm_cell_69/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*=
shared_name.,Adam/lstm_69/lstm_cell_69/recurrent_kernel/m
Ў
@Adam/lstm_69/lstm_cell_69/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_69/lstm_cell_69/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0

 Adam/lstm_69/lstm_cell_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_69/lstm_cell_69/bias/m

4Adam/lstm_69/lstm_cell_69/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_69/lstm_cell_69/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_327/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *(
shared_nameAdam/dense_327/kernel/v

+Adam/dense_327/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_327/kernel/v*
_output_shapes

:2 *
dtype0

Adam/dense_327/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_327/bias/v
{
)Adam/dense_327/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_327/bias/v*
_output_shapes
: *
dtype0

Adam/dense_328/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*(
shared_nameAdam/dense_328/kernel/v

+Adam/dense_328/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_328/kernel/v*
_output_shapes

:2@*
dtype0

Adam/dense_328/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_328/bias/v
{
)Adam/dense_328/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_328/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_329/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*(
shared_nameAdam/dense_329/kernel/v

+Adam/dense_329/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_329/kernel/v*
_output_shapes

:`*
dtype0

Adam/dense_329/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_329/bias/v
{
)Adam/dense_329/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_329/bias/v*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_68/lstm_cell_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*3
shared_name$"Adam/lstm_68/lstm_cell_68/kernel/v

6Adam/lstm_68/lstm_cell_68/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_68/lstm_cell_68/kernel/v*
_output_shapes
:	Ќ*
dtype0
Е
,Adam/lstm_68/lstm_cell_68/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*=
shared_name.,Adam/lstm_68/lstm_cell_68/recurrent_kernel/v
Ў
@Adam/lstm_68/lstm_cell_68/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_68/lstm_cell_68/recurrent_kernel/v*
_output_shapes
:	KЌ*
dtype0

 Adam/lstm_68/lstm_cell_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*1
shared_name" Adam/lstm_68/lstm_cell_68/bias/v

4Adam/lstm_68/lstm_cell_68/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_68/lstm_cell_68/bias/v*
_output_shapes	
:Ќ*
dtype0

 Adam/gru_59/gru_cell_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/gru_59/gru_cell_59/kernel/v

4Adam/gru_59/gru_cell_59/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_59/gru_cell_59/kernel/v*
_output_shapes
:	*
dtype0
Б
*Adam/gru_59/gru_cell_59/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*;
shared_name,*Adam/gru_59/gru_cell_59/recurrent_kernel/v
Њ
>Adam/gru_59/gru_cell_59/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_59/gru_cell_59/recurrent_kernel/v*
_output_shapes
:	2*
dtype0

Adam/gru_59/gru_cell_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/gru_59/gru_cell_59/bias/v

2Adam/gru_59/gru_cell_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_59/gru_cell_59/bias/v*
_output_shapes
:	*
dtype0
Ё
"Adam/lstm_69/lstm_cell_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*3
shared_name$"Adam/lstm_69/lstm_cell_69/kernel/v

6Adam/lstm_69/lstm_cell_69/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_69/lstm_cell_69/kernel/v*
_output_shapes
:	KШ*
dtype0
Е
,Adam/lstm_69/lstm_cell_69/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*=
shared_name.,Adam/lstm_69/lstm_cell_69/recurrent_kernel/v
Ў
@Adam/lstm_69/lstm_cell_69/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_69/lstm_cell_69/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0

 Adam/lstm_69/lstm_cell_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_69/lstm_cell_69/bias/v

4Adam/lstm_69/lstm_cell_69/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_69/lstm_cell_69/bias/v*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
Ї^
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*т]
valueи]Bе] BЮ]

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
 	keras_api
l
!cell
"
state_spec
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
ь
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate+mА,mБ1mВ2mГ;mД<mЕFmЖGmЗHmИImЙJmКKmЛLmМMmНNmО+vП,vР1vС2vТ;vУ<vФFvХGvЦHvЧIvШJvЩKvЪLvЫMvЬNvЭ
n
F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14
n
F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14
 
­
trainable_variables
Olayer_metrics
	variables
Pmetrics

Qlayers
Rlayer_regularization_losses
regularization_losses
Snon_trainable_variables
 
~

Fkernel
Grecurrent_kernel
Hbias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
 

F0
G1
H2

F0
G1
H2
 
Й
trainable_variables
Xlayer_metrics
	variables

Ystates
Zmetrics

[layers
\layer_regularization_losses
regularization_losses
]non_trainable_variables
 
 
 
­
trainable_variables
^layer_metrics
	variables
_metrics

`layers
alayer_regularization_losses
regularization_losses
bnon_trainable_variables
~

Ikernel
Jrecurrent_kernel
Kbias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
 

I0
J1
K2

I0
J1
K2
 
Й
trainable_variables
glayer_metrics
	variables

hstates
imetrics

jlayers
klayer_regularization_losses
regularization_losses
lnon_trainable_variables
~

Lkernel
Mrecurrent_kernel
Nbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
 

L0
M1
N2

L0
M1
N2
 
Й
#trainable_variables
qlayer_metrics
$	variables

rstates
smetrics

tlayers
ulayer_regularization_losses
%regularization_losses
vnon_trainable_variables
 
 
 
­
'trainable_variables
wlayer_metrics
(	variables
xmetrics

ylayers
zlayer_regularization_losses
)regularization_losses
{non_trainable_variables
\Z
VARIABLE_VALUEdense_327/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_327/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
Ў
-trainable_variables
|layer_metrics
.	variables
}metrics

~layers
layer_regularization_losses
/regularization_losses
non_trainable_variables
\Z
VARIABLE_VALUEdense_328/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_328/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
В
3trainable_variables
layer_metrics
4	variables
metrics
layers
 layer_regularization_losses
5regularization_losses
non_trainable_variables
 
 
 
В
7trainable_variables
layer_metrics
8	variables
metrics
layers
 layer_regularization_losses
9regularization_losses
non_trainable_variables
\Z
VARIABLE_VALUEdense_329/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_329/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
В
=trainable_variables
layer_metrics
>	variables
metrics
layers
 layer_regularization_losses
?regularization_losses
non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_68/lstm_cell_68/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_68/lstm_cell_68/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_68/lstm_cell_68/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_59/gru_cell_59/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_59/gru_cell_59/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_59/gru_cell_59/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_69/lstm_cell_69/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_69/lstm_cell_69/recurrent_kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_69/lstm_cell_69/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
F
0
1
2
3
4
5
6
7
	8

9
 
 

F0
G1
H2

F0
G1
H2
 
В
Ttrainable_variables
layer_metrics
U	variables
metrics
layers
 layer_regularization_losses
Vregularization_losses
non_trainable_variables
 
 
 

0
 
 
 
 
 
 
 

I0
J1
K2

I0
J1
K2
 
В
ctrainable_variables
layer_metrics
d	variables
metrics
layers
 layer_regularization_losses
eregularization_losses
non_trainable_variables
 
 
 

0
 
 

L0
M1
N2

L0
M1
N2
 
В
mtrainable_variables
layer_metrics
n	variables
metrics
layers
  layer_regularization_losses
oregularization_losses
Ёnon_trainable_variables
 
 
 

!0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Ђtotal

Ѓcount
Є	variables
Ѕ	keras_api
I

Іtotal

Їcount
Ј
_fn_kwargs
Љ	variables
Њ	keras_api
I

Ћtotal

Ќcount
­
_fn_kwargs
Ў	variables
Џ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ђ0
Ѓ1

Є	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

І0
Ї1

Љ	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ћ0
Ќ1

Ў	variables
}
VARIABLE_VALUEAdam/dense_327/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_327/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_328/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_328/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_329/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_329/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_68/lstm_cell_68/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_68/lstm_cell_68/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_68/lstm_cell_68/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/gru_59/gru_cell_59/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/gru_59/gru_cell_59/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/gru_59/gru_cell_59/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_69/lstm_cell_69/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_69/lstm_cell_69/recurrent_kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_69/lstm_cell_69/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_327/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_327/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_328/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_328/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_329/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_329/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_68/lstm_cell_68/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_68/lstm_cell_68/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_68/lstm_cell_68/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/gru_59/gru_cell_59/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/gru_59/gru_cell_59/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/gru_59/gru_cell_59/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_69/lstm_cell_69/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_69/lstm_cell_69/recurrent_kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_69/lstm_cell_69/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_10Placeholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10lstm_68/lstm_cell_68/kernel%lstm_68/lstm_cell_68/recurrent_kernellstm_68/lstm_cell_68/biasgru_59/gru_cell_59/biasgru_59/gru_cell_59/kernel#gru_59/gru_cell_59/recurrent_kernellstm_69/lstm_cell_69/kernel%lstm_69/lstm_cell_69/recurrent_kernellstm_69/lstm_cell_69/biasdense_327/kerneldense_327/biasdense_328/kerneldense_328/biasdense_329/kerneldense_329/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_50187511
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_327/kernel/Read/ReadVariableOp"dense_327/bias/Read/ReadVariableOp$dense_328/kernel/Read/ReadVariableOp"dense_328/bias/Read/ReadVariableOp$dense_329/kernel/Read/ReadVariableOp"dense_329/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_68/lstm_cell_68/kernel/Read/ReadVariableOp9lstm_68/lstm_cell_68/recurrent_kernel/Read/ReadVariableOp-lstm_68/lstm_cell_68/bias/Read/ReadVariableOp-gru_59/gru_cell_59/kernel/Read/ReadVariableOp7gru_59/gru_cell_59/recurrent_kernel/Read/ReadVariableOp+gru_59/gru_cell_59/bias/Read/ReadVariableOp/lstm_69/lstm_cell_69/kernel/Read/ReadVariableOp9lstm_69/lstm_cell_69/recurrent_kernel/Read/ReadVariableOp-lstm_69/lstm_cell_69/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_327/kernel/m/Read/ReadVariableOp)Adam/dense_327/bias/m/Read/ReadVariableOp+Adam/dense_328/kernel/m/Read/ReadVariableOp)Adam/dense_328/bias/m/Read/ReadVariableOp+Adam/dense_329/kernel/m/Read/ReadVariableOp)Adam/dense_329/bias/m/Read/ReadVariableOp6Adam/lstm_68/lstm_cell_68/kernel/m/Read/ReadVariableOp@Adam/lstm_68/lstm_cell_68/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_68/lstm_cell_68/bias/m/Read/ReadVariableOp4Adam/gru_59/gru_cell_59/kernel/m/Read/ReadVariableOp>Adam/gru_59/gru_cell_59/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_59/gru_cell_59/bias/m/Read/ReadVariableOp6Adam/lstm_69/lstm_cell_69/kernel/m/Read/ReadVariableOp@Adam/lstm_69/lstm_cell_69/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_69/lstm_cell_69/bias/m/Read/ReadVariableOp+Adam/dense_327/kernel/v/Read/ReadVariableOp)Adam/dense_327/bias/v/Read/ReadVariableOp+Adam/dense_328/kernel/v/Read/ReadVariableOp)Adam/dense_328/bias/v/Read/ReadVariableOp+Adam/dense_329/kernel/v/Read/ReadVariableOp)Adam/dense_329/bias/v/Read/ReadVariableOp6Adam/lstm_68/lstm_cell_68/kernel/v/Read/ReadVariableOp@Adam/lstm_68/lstm_cell_68/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_68/lstm_cell_68/bias/v/Read/ReadVariableOp4Adam/gru_59/gru_cell_59/kernel/v/Read/ReadVariableOp>Adam/gru_59/gru_cell_59/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_59/gru_cell_59/bias/v/Read/ReadVariableOp6Adam/lstm_69/lstm_cell_69/kernel/v/Read/ReadVariableOp@Adam/lstm_69/lstm_cell_69/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_69/lstm_cell_69/bias/v/Read/ReadVariableOpConst*E
Tin>
<2:	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_50191174

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_327/kerneldense_327/biasdense_328/kerneldense_328/biasdense_329/kerneldense_329/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_68/lstm_cell_68/kernel%lstm_68/lstm_cell_68/recurrent_kernellstm_68/lstm_cell_68/biasgru_59/gru_cell_59/kernel#gru_59/gru_cell_59/recurrent_kernelgru_59/gru_cell_59/biaslstm_69/lstm_cell_69/kernel%lstm_69/lstm_cell_69/recurrent_kernellstm_69/lstm_cell_69/biastotalcounttotal_1count_1total_2count_2Adam/dense_327/kernel/mAdam/dense_327/bias/mAdam/dense_328/kernel/mAdam/dense_328/bias/mAdam/dense_329/kernel/mAdam/dense_329/bias/m"Adam/lstm_68/lstm_cell_68/kernel/m,Adam/lstm_68/lstm_cell_68/recurrent_kernel/m Adam/lstm_68/lstm_cell_68/bias/m Adam/gru_59/gru_cell_59/kernel/m*Adam/gru_59/gru_cell_59/recurrent_kernel/mAdam/gru_59/gru_cell_59/bias/m"Adam/lstm_69/lstm_cell_69/kernel/m,Adam/lstm_69/lstm_cell_69/recurrent_kernel/m Adam/lstm_69/lstm_cell_69/bias/mAdam/dense_327/kernel/vAdam/dense_327/bias/vAdam/dense_328/kernel/vAdam/dense_328/bias/vAdam/dense_329/kernel/vAdam/dense_329/bias/v"Adam/lstm_68/lstm_cell_68/kernel/v,Adam/lstm_68/lstm_cell_68/recurrent_kernel/v Adam/lstm_68/lstm_cell_68/bias/v Adam/gru_59/gru_cell_59/kernel/v*Adam/gru_59/gru_cell_59/recurrent_kernel/vAdam/gru_59/gru_cell_59/bias/v"Adam/lstm_69/lstm_cell_69/kernel/v,Adam/lstm_69/lstm_cell_69/recurrent_kernel/v Adam/lstm_69/lstm_cell_69/bias/v*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_50191352№6
Е
п
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190741

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
ё	
р
G__inference_dense_327_layer_call_and_return_conditional_losses_50190614

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
O


lstm_68_while_body_50187579,
(lstm_68_while_lstm_68_while_loop_counter2
.lstm_68_while_lstm_68_while_maximum_iterations
lstm_68_while_placeholder
lstm_68_while_placeholder_1
lstm_68_while_placeholder_2
lstm_68_while_placeholder_3+
'lstm_68_while_lstm_68_strided_slice_1_0g
clstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0A
=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0@
<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0
lstm_68_while_identity
lstm_68_while_identity_1
lstm_68_while_identity_2
lstm_68_while_identity_3
lstm_68_while_identity_4
lstm_68_while_identity_5)
%lstm_68_while_lstm_68_strided_slice_1e
alstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor=
9lstm_68_while_lstm_cell_68_matmul_readvariableop_resource?
;lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource>
:lstm_68_while_lstm_cell_68_biasadd_readvariableop_resourceЂ1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpЂ0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpЂ2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpг
?lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_68/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0lstm_68_while_placeholderHlstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_68/while/TensorArrayV2Read/TensorListGetItemс
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype022
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpї
!lstm_68/while/lstm_cell_68/MatMulMatMul8lstm_68/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!lstm_68/while/lstm_cell_68/MatMulч
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype024
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpр
#lstm_68/while/lstm_cell_68/MatMul_1MatMullstm_68_while_placeholder_2:lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#lstm_68/while/lstm_cell_68/MatMul_1и
lstm_68/while/lstm_cell_68/addAddV2+lstm_68/while/lstm_cell_68/MatMul:product:0-lstm_68/while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm_68/while/lstm_cell_68/addр
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype023
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpх
"lstm_68/while/lstm_cell_68/BiasAddBiasAdd"lstm_68/while/lstm_cell_68/add:z:09lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_68/while/lstm_cell_68/BiasAdd
 lstm_68/while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_68/while/lstm_cell_68/Const
*lstm_68/while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_68/while/lstm_cell_68/split/split_dimЋ
 lstm_68/while/lstm_cell_68/splitSplit3lstm_68/while/lstm_cell_68/split/split_dim:output:0+lstm_68/while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2"
 lstm_68/while/lstm_cell_68/splitА
"lstm_68/while/lstm_cell_68/SigmoidSigmoid)lstm_68/while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"lstm_68/while/lstm_cell_68/SigmoidД
$lstm_68/while/lstm_cell_68/Sigmoid_1Sigmoid)lstm_68/while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_68/while/lstm_cell_68/Sigmoid_1Р
lstm_68/while/lstm_cell_68/mulMul(lstm_68/while/lstm_cell_68/Sigmoid_1:y:0lstm_68_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/while/lstm_cell_68/mulЇ
lstm_68/while/lstm_cell_68/ReluRelu)lstm_68/while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2!
lstm_68/while/lstm_cell_68/Reluд
 lstm_68/while/lstm_cell_68/mul_1Mul&lstm_68/while/lstm_cell_68/Sigmoid:y:0-lstm_68/while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/mul_1Щ
 lstm_68/while/lstm_cell_68/add_1AddV2"lstm_68/while/lstm_cell_68/mul:z:0$lstm_68/while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/add_1Д
$lstm_68/while/lstm_cell_68/Sigmoid_2Sigmoid)lstm_68/while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_68/while/lstm_cell_68/Sigmoid_2І
!lstm_68/while/lstm_cell_68/Relu_1Relu$lstm_68/while/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2#
!lstm_68/while/lstm_cell_68/Relu_1и
 lstm_68/while/lstm_cell_68/mul_2Mul(lstm_68/while/lstm_cell_68/Sigmoid_2:y:0/lstm_68/while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/mul_2
2lstm_68/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_68_while_placeholder_1lstm_68_while_placeholder$lstm_68/while/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_68/while/TensorArrayV2Write/TensorListSetIteml
lstm_68/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/while/add/y
lstm_68/while/addAddV2lstm_68_while_placeholderlstm_68/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_68/while/addp
lstm_68/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/while/add_1/y
lstm_68/while/add_1AddV2(lstm_68_while_lstm_68_while_loop_counterlstm_68/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_68/while/add_1
lstm_68/while/IdentityIdentitylstm_68/while/add_1:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity­
lstm_68/while/Identity_1Identity.lstm_68_while_lstm_68_while_maximum_iterations2^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_1
lstm_68/while/Identity_2Identitylstm_68/while/add:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_2С
lstm_68/while/Identity_3IdentityBlstm_68/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_3Д
lstm_68/while/Identity_4Identity$lstm_68/while/lstm_cell_68/mul_2:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/while/Identity_4Д
lstm_68/while/Identity_5Identity$lstm_68/while/lstm_cell_68/add_1:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/while/Identity_5"9
lstm_68_while_identitylstm_68/while/Identity:output:0"=
lstm_68_while_identity_1!lstm_68/while/Identity_1:output:0"=
lstm_68_while_identity_2!lstm_68/while/Identity_2:output:0"=
lstm_68_while_identity_3!lstm_68/while/Identity_3:output:0"=
lstm_68_while_identity_4!lstm_68/while/Identity_4:output:0"=
lstm_68_while_identity_5!lstm_68/while/Identity_5:output:0"P
%lstm_68_while_lstm_68_strided_slice_1'lstm_68_while_lstm_68_strided_slice_1_0"z
:lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0"|
;lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0"x
9lstm_68_while_lstm_cell_68_matmul_readvariableop_resource;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0"Ш
alstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensorclstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2f
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp2d
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp2h
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
в[
н
D__inference_gru_59_layer_call_and_return_conditional_losses_50186744

inputs'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50186654*
condR
while_cond_50186653*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
в[
н
D__inference_gru_59_layer_call_and_return_conditional_losses_50189399

inputs'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50189309*
condR
while_cond_50189308*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
C

while_body_50188953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
М

*__inference_lstm_68_layer_call_fn_50188885
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501849072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
њG
А
while_body_50189649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
C

while_body_50186159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_50184837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50184837___redundant_placeholder06
2while_while_cond_50184837___redundant_placeholder16
2while_while_cond_50184837___redundant_placeholder26
2while_while_cond_50184837___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
Е
Э
while_cond_50188624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50188624___redundant_placeholder06
2while_while_cond_50188624___redundant_placeholder16
2while_while_cond_50188624___redundant_placeholder26
2while_while_cond_50188624___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
Щ.
в
E__inference_model_9_layer_call_and_return_conditional_losses_50187355

inputs
lstm_68_50187315
lstm_68_50187317
lstm_68_50187319
gru_59_50187322
gru_59_50187324
gru_59_50187326
lstm_69_50187331
lstm_69_50187333
lstm_69_50187335
dense_327_50187338
dense_327_50187340
dense_328_50187343
dense_328_50187345
dense_329_50187349
dense_329_50187351
identityЂ!dense_327/StatefulPartitionedCallЂ!dense_328/StatefulPartitionedCallЂ!dense_329/StatefulPartitionedCallЂ"dropout_68/StatefulPartitionedCallЂ"dropout_69/StatefulPartitionedCallЂgru_59/StatefulPartitionedCallЂlstm_68/StatefulPartitionedCallЂlstm_69/StatefulPartitionedCallЙ
lstm_68/StatefulPartitionedCallStatefulPartitionedCallinputslstm_68_50187315lstm_68_50187317lstm_68_50187319*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501862442!
lstm_68/StatefulPartitionedCallІ
gru_59/StatefulPartitionedCallStatefulPartitionedCallinputsgru_59_50187322gru_59_50187324gru_59_50187326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501865852 
gru_59/StatefulPartitionedCallІ
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall(lstm_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867862$
"dropout_68/StatefulPartitionedCallН
"dropout_69/StatefulPartitionedCallStatefulPartitionedCall'gru_59/StatefulPartitionedCall:output:0#^dropout_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868162$
"dropout_69/StatefulPartitionedCallб
lstm_69/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0lstm_69_50187331lstm_69_50187333lstm_69_50187335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501869862!
lstm_69/StatefulPartitionedCallФ
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(lstm_69/StatefulPartitionedCall:output:0dense_327_50187338dense_327_50187340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_327_layer_call_and_return_conditional_losses_501871802#
!dense_327/StatefulPartitionedCallЧ
!dense_328/StatefulPartitionedCallStatefulPartitionedCall+dropout_69/StatefulPartitionedCall:output:0dense_328_50187343dense_328_50187345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_328_layer_call_and_return_conditional_losses_501872072#
!dense_328/StatefulPartitionedCallЙ
concatenate_9/PartitionedCallPartitionedCall*dense_327/StatefulPartitionedCall:output:0*dense_328/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_501872302
concatenate_9/PartitionedCallТ
!dense_329/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_329_50187349dense_329_50187351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_329_layer_call_and_return_conditional_losses_501872492#
!dense_329/StatefulPartitionedCall
IdentityIdentity*dense_329/StatefulPartitionedCall:output:0"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall#^dropout_69/StatefulPartitionedCall^gru_59/StatefulPartitionedCall ^lstm_68/StatefulPartitionedCall ^lstm_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2H
"dropout_69/StatefulPartitionedCall"dropout_69/StatefulPartitionedCall2@
gru_59/StatefulPartitionedCallgru_59/StatefulPartitionedCall2B
lstm_68/StatefulPartitionedCalllstm_68/StatefulPartitionedCall2B
lstm_69/StatefulPartitionedCalllstm_69/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ
f
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189230

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
њG
А
while_body_50189309
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ђ

*__inference_lstm_69_layer_call_fn_50190237
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501859472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
Щ[
є
E__inference_lstm_68_layer_call_and_return_conditional_losses_50186397

inputs/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50186312*
condR
while_cond_50186311*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
в[
н
D__inference_gru_59_layer_call_and_return_conditional_losses_50186585

inputs'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50186495*
condR
while_cond_50186494*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т
Я
/__inference_lstm_cell_69_layer_call_fn_50190966

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
O


lstm_69_while_body_50188380,
(lstm_69_while_lstm_69_while_loop_counter2
.lstm_69_while_lstm_69_while_maximum_iterations
lstm_69_while_placeholder
lstm_69_while_placeholder_1
lstm_69_while_placeholder_2
lstm_69_while_placeholder_3+
'lstm_69_while_lstm_69_strided_slice_1_0g
clstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0A
=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0@
<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0
lstm_69_while_identity
lstm_69_while_identity_1
lstm_69_while_identity_2
lstm_69_while_identity_3
lstm_69_while_identity_4
lstm_69_while_identity_5)
%lstm_69_while_lstm_69_strided_slice_1e
alstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor=
9lstm_69_while_lstm_cell_69_matmul_readvariableop_resource?
;lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource>
:lstm_69_while_lstm_cell_69_biasadd_readvariableop_resourceЂ1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpЂ0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpЂ2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpг
?lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2A
?lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_69/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0lstm_69_while_placeholderHlstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype023
1lstm_69/while/TensorArrayV2Read/TensorListGetItemс
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype022
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpї
!lstm_69/while/lstm_cell_69/MatMulMatMul8lstm_69/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!lstm_69/while/lstm_cell_69/MatMulч
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype024
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpр
#lstm_69/while/lstm_cell_69/MatMul_1MatMullstm_69_while_placeholder_2:lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#lstm_69/while/lstm_cell_69/MatMul_1и
lstm_69/while/lstm_cell_69/addAddV2+lstm_69/while/lstm_cell_69/MatMul:product:0-lstm_69/while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
lstm_69/while/lstm_cell_69/addр
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpх
"lstm_69/while/lstm_cell_69/BiasAddBiasAdd"lstm_69/while/lstm_cell_69/add:z:09lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"lstm_69/while/lstm_cell_69/BiasAdd
 lstm_69/while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_69/while/lstm_cell_69/Const
*lstm_69/while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_69/while/lstm_cell_69/split/split_dimЋ
 lstm_69/while/lstm_cell_69/splitSplit3lstm_69/while/lstm_cell_69/split/split_dim:output:0+lstm_69/while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 lstm_69/while/lstm_cell_69/splitА
"lstm_69/while/lstm_cell_69/SigmoidSigmoid)lstm_69/while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"lstm_69/while/lstm_cell_69/SigmoidД
$lstm_69/while/lstm_cell_69/Sigmoid_1Sigmoid)lstm_69/while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_69/while/lstm_cell_69/Sigmoid_1Р
lstm_69/while/lstm_cell_69/mulMul(lstm_69/while/lstm_cell_69/Sigmoid_1:y:0lstm_69_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/while/lstm_cell_69/mulЇ
lstm_69/while/lstm_cell_69/ReluRelu)lstm_69/while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
lstm_69/while/lstm_cell_69/Reluд
 lstm_69/while/lstm_cell_69/mul_1Mul&lstm_69/while/lstm_cell_69/Sigmoid:y:0-lstm_69/while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/mul_1Щ
 lstm_69/while/lstm_cell_69/add_1AddV2"lstm_69/while/lstm_cell_69/mul:z:0$lstm_69/while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/add_1Д
$lstm_69/while/lstm_cell_69/Sigmoid_2Sigmoid)lstm_69/while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_69/while/lstm_cell_69/Sigmoid_2І
!lstm_69/while/lstm_cell_69/Relu_1Relu$lstm_69/while/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!lstm_69/while/lstm_cell_69/Relu_1и
 lstm_69/while/lstm_cell_69/mul_2Mul(lstm_69/while/lstm_cell_69/Sigmoid_2:y:0/lstm_69/while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/mul_2
2lstm_69/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_69_while_placeholder_1lstm_69_while_placeholder$lstm_69/while/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_69/while/TensorArrayV2Write/TensorListSetIteml
lstm_69/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/while/add/y
lstm_69/while/addAddV2lstm_69_while_placeholderlstm_69/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_69/while/addp
lstm_69/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/while/add_1/y
lstm_69/while/add_1AddV2(lstm_69_while_lstm_69_while_loop_counterlstm_69/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_69/while/add_1
lstm_69/while/IdentityIdentitylstm_69/while/add_1:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity­
lstm_69/while/Identity_1Identity.lstm_69_while_lstm_69_while_maximum_iterations2^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_1
lstm_69/while/Identity_2Identitylstm_69/while/add:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_2С
lstm_69/while/Identity_3IdentityBlstm_69/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_3Д
lstm_69/while/Identity_4Identity$lstm_69/while/lstm_cell_69/mul_2:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/while/Identity_4Д
lstm_69/while/Identity_5Identity$lstm_69/while/lstm_cell_69/add_1:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/while/Identity_5"9
lstm_69_while_identitylstm_69/while/Identity:output:0"=
lstm_69_while_identity_1!lstm_69/while/Identity_1:output:0"=
lstm_69_while_identity_2!lstm_69/while/Identity_2:output:0"=
lstm_69_while_identity_3!lstm_69/while/Identity_3:output:0"=
lstm_69_while_identity_4!lstm_69/while/Identity_4:output:0"=
lstm_69_while_identity_5!lstm_69/while/Identity_5:output:0"P
%lstm_69_while_lstm_69_strided_slice_1'lstm_69_while_lstm_69_strided_slice_1_0"z
:lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0"|
;lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0"x
9lstm_69_while_lstm_cell_69_matmul_readvariableop_resource;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0"Ш
alstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensorclstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2f
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp2d
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp2h
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_50189467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50189467___redundant_placeholder06
2while_while_cond_50189467___redundant_placeholder16
2while_while_cond_50189467___redundant_placeholder26
2while_while_cond_50189467___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Е
Э
while_cond_50186009
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50186009___redundant_placeholder06
2while_while_cond_50186009___redundant_placeholder16
2while_while_cond_50186009___redundant_placeholder26
2while_while_cond_50186009___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к[
п
D__inference_gru_59_layer_call_and_return_conditional_losses_50189898
inputs_0'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50189808*
condR
while_cond_50189807*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
C

while_body_50189106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
C

while_body_50188625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Р%

while_body_50186010
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_69_50186034_0!
while_lstm_cell_69_50186036_0!
while_lstm_cell_69_50186038_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_69_50186034
while_lstm_cell_69_50186036
while_lstm_cell_69_50186038Ђ*while/lstm_cell_69/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_69/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_69_50186034_0while_lstm_cell_69_50186036_0while_lstm_cell_69_50186038_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855842,
*while/lstm_cell_69/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_69/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_69/StatefulPartitionedCall:output:1+^while/lstm_cell_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_69/StatefulPartitionedCall:output:2+^while/lstm_cell_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_69_50186034while_lstm_cell_69_50186034_0"<
while_lstm_cell_69_50186036while_lstm_cell_69_50186036_0"<
while_lstm_cell_69_50186038while_lstm_cell_69_50186038_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2X
*while/lstm_cell_69/StatefulPartitionedCall*while/lstm_cell_69/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
д
А
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190815

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
Х[
є
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190554

inputs/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50190469*
condR
while_cond_50190468*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
=
п
D__inference_gru_59_layer_call_and_return_conditional_losses_50185469

inputs
gru_cell_59_50185393
gru_cell_59_50185395
gru_cell_59_50185397
identityЂ#gru_cell_59/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2ћ
#gru_cell_59/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_59_50185393gru_cell_59_50185395gru_cell_59_50185397*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501850282%
#gru_cell_59/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_59_50185393gru_cell_59_50185395gru_cell_59_50185397*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50185405*
condR
while_cond_50185404*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^gru_cell_59/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_59/StatefulPartitionedCall#gru_cell_59/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ф+

E__inference_model_9_layer_call_and_return_conditional_losses_50187309
input_10
lstm_68_50187269
lstm_68_50187271
lstm_68_50187273
gru_59_50187276
gru_59_50187278
gru_59_50187280
lstm_69_50187285
lstm_69_50187287
lstm_69_50187289
dense_327_50187292
dense_327_50187294
dense_328_50187297
dense_328_50187299
dense_329_50187303
dense_329_50187305
identityЂ!dense_327/StatefulPartitionedCallЂ!dense_328/StatefulPartitionedCallЂ!dense_329/StatefulPartitionedCallЂgru_59/StatefulPartitionedCallЂlstm_68/StatefulPartitionedCallЂlstm_69/StatefulPartitionedCallЛ
lstm_68/StatefulPartitionedCallStatefulPartitionedCallinput_10lstm_68_50187269lstm_68_50187271lstm_68_50187273*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501863972!
lstm_68/StatefulPartitionedCallЈ
gru_59/StatefulPartitionedCallStatefulPartitionedCallinput_10gru_59_50187276gru_59_50187278gru_59_50187280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501867442 
gru_59/StatefulPartitionedCall
dropout_68/PartitionedCallPartitionedCall(lstm_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867912
dropout_68/PartitionedCall
dropout_69/PartitionedCallPartitionedCall'gru_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868212
dropout_69/PartitionedCallЩ
lstm_69/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0lstm_69_50187285lstm_69_50187287lstm_69_50187289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501871392!
lstm_69/StatefulPartitionedCallФ
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(lstm_69/StatefulPartitionedCall:output:0dense_327_50187292dense_327_50187294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_327_layer_call_and_return_conditional_losses_501871802#
!dense_327/StatefulPartitionedCallП
!dense_328/StatefulPartitionedCallStatefulPartitionedCall#dropout_69/PartitionedCall:output:0dense_328_50187297dense_328_50187299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_328_layer_call_and_return_conditional_losses_501872072#
!dense_328/StatefulPartitionedCallЙ
concatenate_9/PartitionedCallPartitionedCall*dense_327/StatefulPartitionedCall:output:0*dense_328/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_501872302
concatenate_9/PartitionedCallТ
!dense_329/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_329_50187303dense_329_50187305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_329_layer_call_and_return_conditional_losses_501872492#
!dense_329/StatefulPartitionedCallЯ
IdentityIdentity*dense_329/StatefulPartitionedCall:output:0"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall^gru_59/StatefulPartitionedCall ^lstm_68/StatefulPartitionedCall ^lstm_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2@
gru_59/StatefulPartitionedCallgru_59/StatefulPartitionedCall2B
lstm_68/StatefulPartitionedCalllstm_68/StatefulPartitionedCall2B
lstm_69/StatefulPartitionedCalllstm_69/StatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Р
w
K__inference_concatenate_9_layer_call_and_return_conditional_losses_50190650
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1
Ј
\
0__inference_concatenate_9_layer_call_fn_50190656
inputs_0
inputs_1
identityй
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_501872302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1
Љ
f
-__inference_dropout_69_layer_call_fn_50190598

inputs
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
б
I
-__inference_dropout_68_layer_call_fn_50189240

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867912
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
S
	
gru_59_while_body_50187729*
&gru_59_while_gru_59_while_loop_counter0
,gru_59_while_gru_59_while_maximum_iterations
gru_59_while_placeholder
gru_59_while_placeholder_1
gru_59_while_placeholder_2)
%gru_59_while_gru_59_strided_slice_1_0e
agru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_06
2gru_59_while_gru_cell_59_readvariableop_resource_0=
9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0?
;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0
gru_59_while_identity
gru_59_while_identity_1
gru_59_while_identity_2
gru_59_while_identity_3
gru_59_while_identity_4'
#gru_59_while_gru_59_strided_slice_1c
_gru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor4
0gru_59_while_gru_cell_59_readvariableop_resource;
7gru_59_while_gru_cell_59_matmul_readvariableop_resource=
9gru_59_while_gru_cell_59_matmul_1_readvariableop_resourceЂ.gru_59/while/gru_cell_59/MatMul/ReadVariableOpЂ0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpЂ'gru_59/while/gru_cell_59/ReadVariableOpб
>gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0gru_59/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_0gru_59_while_placeholderGgru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype022
0gru_59/while/TensorArrayV2Read/TensorListGetItemЦ
'gru_59/while/gru_cell_59/ReadVariableOpReadVariableOp2gru_59_while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'gru_59/while/gru_cell_59/ReadVariableOpЗ
 gru_59/while/gru_cell_59/unstackUnpack/gru_59/while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2"
 gru_59/while/gru_cell_59/unstackл
.gru_59/while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.gru_59/while/gru_cell_59/MatMul/ReadVariableOp№
gru_59/while/gru_cell_59/MatMulMatMul7gru_59/while/TensorArrayV2Read/TensorListGetItem:item:06gru_59/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
gru_59/while/gru_cell_59/MatMulи
 gru_59/while/gru_cell_59/BiasAddBiasAdd)gru_59/while/gru_cell_59/MatMul:product:0)gru_59/while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 gru_59/while/gru_cell_59/BiasAdd
gru_59/while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_59/while/gru_cell_59/Const
(gru_59/while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(gru_59/while/gru_cell_59/split/split_dim
gru_59/while/gru_cell_59/splitSplit1gru_59/while/gru_cell_59/split/split_dim:output:0)gru_59/while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2 
gru_59/while/gru_cell_59/splitс
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype022
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpй
!gru_59/while/gru_cell_59/MatMul_1MatMulgru_59_while_placeholder_28gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2#
!gru_59/while/gru_cell_59/MatMul_1о
"gru_59/while/gru_cell_59/BiasAdd_1BiasAdd+gru_59/while/gru_cell_59/MatMul_1:product:0)gru_59/while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2$
"gru_59/while/gru_cell_59/BiasAdd_1
 gru_59/while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2"
 gru_59/while/gru_cell_59/Const_1Ѓ
*gru_59/while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*gru_59/while/gru_cell_59/split_1/split_dimа
 gru_59/while/gru_cell_59/split_1SplitV+gru_59/while/gru_cell_59/BiasAdd_1:output:0)gru_59/while/gru_cell_59/Const_1:output:03gru_59/while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 gru_59/while/gru_cell_59/split_1Ы
gru_59/while/gru_cell_59/addAddV2'gru_59/while/gru_cell_59/split:output:0)gru_59/while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/addЃ
 gru_59/while/gru_cell_59/SigmoidSigmoid gru_59/while/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 gru_59/while/gru_cell_59/SigmoidЯ
gru_59/while/gru_cell_59/add_1AddV2'gru_59/while/gru_cell_59/split:output:1)gru_59/while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_1Љ
"gru_59/while/gru_cell_59/Sigmoid_1Sigmoid"gru_59/while/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"gru_59/while/gru_cell_59/Sigmoid_1Ш
gru_59/while/gru_cell_59/mulMul&gru_59/while/gru_cell_59/Sigmoid_1:y:0)gru_59/while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/mulЦ
gru_59/while/gru_cell_59/add_2AddV2'gru_59/while/gru_cell_59/split:output:2 gru_59/while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_2
gru_59/while/gru_cell_59/ReluRelu"gru_59/while/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/ReluЛ
gru_59/while/gru_cell_59/mul_1Mul$gru_59/while/gru_cell_59/Sigmoid:y:0gru_59_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/mul_1
gru_59/while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru_59/while/gru_cell_59/sub/xФ
gru_59/while/gru_cell_59/subSub'gru_59/while/gru_cell_59/sub/x:output:0$gru_59/while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/subШ
gru_59/while/gru_cell_59/mul_2Mul gru_59/while/gru_cell_59/sub:z:0+gru_59/while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/mul_2У
gru_59/while/gru_cell_59/add_3AddV2"gru_59/while/gru_cell_59/mul_1:z:0"gru_59/while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_3
1gru_59/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_59_while_placeholder_1gru_59_while_placeholder"gru_59/while/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_59/while/TensorArrayV2Write/TensorListSetItemj
gru_59/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/while/add/y
gru_59/while/addAddV2gru_59_while_placeholdergru_59/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_59/while/addn
gru_59/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/while/add_1/y
gru_59/while/add_1AddV2&gru_59_while_gru_59_while_loop_countergru_59/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_59/while/add_1
gru_59/while/IdentityIdentitygru_59/while/add_1:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity
gru_59/while/Identity_1Identity,gru_59_while_gru_59_while_maximum_iterations/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_1
gru_59/while/Identity_2Identitygru_59/while/add:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_2А
gru_59/while/Identity_3IdentityAgru_59/while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_3Ђ
gru_59/while/Identity_4Identity"gru_59/while/gru_cell_59/add_3:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/Identity_4"L
#gru_59_while_gru_59_strided_slice_1%gru_59_while_gru_59_strided_slice_1_0"x
9gru_59_while_gru_cell_59_matmul_1_readvariableop_resource;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0"t
7gru_59_while_gru_cell_59_matmul_readvariableop_resource9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0"f
0gru_59_while_gru_cell_59_readvariableop_resource2gru_59_while_gru_cell_59_readvariableop_resource_0"7
gru_59_while_identitygru_59/while/Identity:output:0";
gru_59_while_identity_1 gru_59/while/Identity_1:output:0";
gru_59_while_identity_2 gru_59/while/Identity_2:output:0";
gru_59_while_identity_3 gru_59/while/Identity_3:output:0";
gru_59_while_identity_4 gru_59/while/Identity_4:output:0"Ф
_gru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensoragru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2`
.gru_59/while/gru_cell_59/MatMul/ReadVariableOp.gru_59/while/gru_cell_59/MatMul/ReadVariableOp2d
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp2R
'gru_59/while/gru_cell_59/ReadVariableOp'gru_59/while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ц

,__inference_dense_328_layer_call_fn_50190643

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_328_layer_call_and_return_conditional_losses_501872072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Њ

Ш
*__inference_model_9_layer_call_fn_50187466
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_501874332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
я
э
#__inference__wrapped_model_50184306
input_10?
;model_9_lstm_68_lstm_cell_68_matmul_readvariableop_resourceA
=model_9_lstm_68_lstm_cell_68_matmul_1_readvariableop_resource@
<model_9_lstm_68_lstm_cell_68_biasadd_readvariableop_resource6
2model_9_gru_59_gru_cell_59_readvariableop_resource=
9model_9_gru_59_gru_cell_59_matmul_readvariableop_resource?
;model_9_gru_59_gru_cell_59_matmul_1_readvariableop_resource?
;model_9_lstm_69_lstm_cell_69_matmul_readvariableop_resourceA
=model_9_lstm_69_lstm_cell_69_matmul_1_readvariableop_resource@
<model_9_lstm_69_lstm_cell_69_biasadd_readvariableop_resource4
0model_9_dense_327_matmul_readvariableop_resource5
1model_9_dense_327_biasadd_readvariableop_resource4
0model_9_dense_328_matmul_readvariableop_resource5
1model_9_dense_328_biasadd_readvariableop_resource4
0model_9_dense_329_matmul_readvariableop_resource5
1model_9_dense_329_biasadd_readvariableop_resource
identityЂ(model_9/dense_327/BiasAdd/ReadVariableOpЂ'model_9/dense_327/MatMul/ReadVariableOpЂ(model_9/dense_328/BiasAdd/ReadVariableOpЂ'model_9/dense_328/MatMul/ReadVariableOpЂ(model_9/dense_329/BiasAdd/ReadVariableOpЂ'model_9/dense_329/MatMul/ReadVariableOpЂ0model_9/gru_59/gru_cell_59/MatMul/ReadVariableOpЂ2model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOpЂ)model_9/gru_59/gru_cell_59/ReadVariableOpЂmodel_9/gru_59/whileЂ3model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpЂ2model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOpЂ4model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpЂmodel_9/lstm_68/whileЂ3model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpЂ2model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOpЂ4model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpЂmodel_9/lstm_69/whilef
model_9/lstm_68/ShapeShapeinput_10*
T0*
_output_shapes
:2
model_9/lstm_68/Shape
#model_9/lstm_68/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/lstm_68/strided_slice/stack
%model_9/lstm_68/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_68/strided_slice/stack_1
%model_9/lstm_68/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_68/strided_slice/stack_2Т
model_9/lstm_68/strided_sliceStridedSlicemodel_9/lstm_68/Shape:output:0,model_9/lstm_68/strided_slice/stack:output:0.model_9/lstm_68/strided_slice/stack_1:output:0.model_9/lstm_68/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_68/strided_slice|
model_9/lstm_68/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
model_9/lstm_68/zeros/mul/yЌ
model_9/lstm_68/zeros/mulMul&model_9/lstm_68/strided_slice:output:0$model_9/lstm_68/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/zeros/mul
model_9/lstm_68/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/lstm_68/zeros/Less/yЇ
model_9/lstm_68/zeros/LessLessmodel_9/lstm_68/zeros/mul:z:0%model_9/lstm_68/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/zeros/Less
model_9/lstm_68/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2 
model_9/lstm_68/zeros/packed/1У
model_9/lstm_68/zeros/packedPack&model_9/lstm_68/strided_slice:output:0'model_9/lstm_68/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_68/zeros/packed
model_9/lstm_68/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_68/zeros/ConstЕ
model_9/lstm_68/zerosFill%model_9/lstm_68/zeros/packed:output:0$model_9/lstm_68/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
model_9/lstm_68/zeros
model_9/lstm_68/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
model_9/lstm_68/zeros_1/mul/yВ
model_9/lstm_68/zeros_1/mulMul&model_9/lstm_68/strided_slice:output:0&model_9/lstm_68/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/zeros_1/mul
model_9/lstm_68/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
model_9/lstm_68/zeros_1/Less/yЏ
model_9/lstm_68/zeros_1/LessLessmodel_9/lstm_68/zeros_1/mul:z:0'model_9/lstm_68/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/zeros_1/Less
 model_9/lstm_68/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2"
 model_9/lstm_68/zeros_1/packed/1Щ
model_9/lstm_68/zeros_1/packedPack&model_9/lstm_68/strided_slice:output:0)model_9/lstm_68/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_9/lstm_68/zeros_1/packed
model_9/lstm_68/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_68/zeros_1/ConstН
model_9/lstm_68/zeros_1Fill'model_9/lstm_68/zeros_1/packed:output:0&model_9/lstm_68/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
model_9/lstm_68/zeros_1
model_9/lstm_68/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_9/lstm_68/transpose/permЕ
model_9/lstm_68/transpose	Transposeinput_10'model_9/lstm_68/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
model_9/lstm_68/transpose
model_9/lstm_68/Shape_1Shapemodel_9/lstm_68/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_68/Shape_1
%model_9/lstm_68/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_68/strided_slice_1/stack
'model_9/lstm_68/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_68/strided_slice_1/stack_1
'model_9/lstm_68/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_68/strided_slice_1/stack_2Ю
model_9/lstm_68/strided_slice_1StridedSlice model_9/lstm_68/Shape_1:output:0.model_9/lstm_68/strided_slice_1/stack:output:00model_9/lstm_68/strided_slice_1/stack_1:output:00model_9/lstm_68/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_9/lstm_68/strided_slice_1Ѕ
+model_9/lstm_68/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+model_9/lstm_68/TensorArrayV2/element_shapeђ
model_9/lstm_68/TensorArrayV2TensorListReserve4model_9/lstm_68/TensorArrayV2/element_shape:output:0(model_9/lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_68/TensorArrayV2п
Emodel_9/lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Emodel_9/lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7model_9/lstm_68/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_68/transpose:y:0Nmodel_9/lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_9/lstm_68/TensorArrayUnstack/TensorListFromTensor
%model_9/lstm_68/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_68/strided_slice_2/stack
'model_9/lstm_68/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_68/strided_slice_2/stack_1
'model_9/lstm_68/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_68/strided_slice_2/stack_2м
model_9/lstm_68/strided_slice_2StridedSlicemodel_9/lstm_68/transpose:y:0.model_9/lstm_68/strided_slice_2/stack:output:00model_9/lstm_68/strided_slice_2/stack_1:output:00model_9/lstm_68/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
model_9/lstm_68/strided_slice_2х
2model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp;model_9_lstm_68_lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype024
2model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOpэ
#model_9/lstm_68/lstm_cell_68/MatMulMatMul(model_9/lstm_68/strided_slice_2:output:0:model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#model_9/lstm_68/lstm_cell_68/MatMulы
4model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp=model_9_lstm_68_lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype026
4model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpщ
%model_9/lstm_68/lstm_cell_68/MatMul_1MatMulmodel_9/lstm_68/zeros:output:0<model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%model_9/lstm_68/lstm_cell_68/MatMul_1р
 model_9/lstm_68/lstm_cell_68/addAddV2-model_9/lstm_68/lstm_cell_68/MatMul:product:0/model_9/lstm_68/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 model_9/lstm_68/lstm_cell_68/addф
3model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp<model_9_lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype025
3model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpэ
$model_9/lstm_68/lstm_cell_68/BiasAddBiasAdd$model_9/lstm_68/lstm_cell_68/add:z:0;model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$model_9/lstm_68/lstm_cell_68/BiasAdd
"model_9/lstm_68/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/lstm_68/lstm_cell_68/Const
,model_9/lstm_68/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_9/lstm_68/lstm_cell_68/split/split_dimГ
"model_9/lstm_68/lstm_cell_68/splitSplit5model_9/lstm_68/lstm_cell_68/split/split_dim:output:0-model_9/lstm_68/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2$
"model_9/lstm_68/lstm_cell_68/splitЖ
$model_9/lstm_68/lstm_cell_68/SigmoidSigmoid+model_9/lstm_68/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2&
$model_9/lstm_68/lstm_cell_68/SigmoidК
&model_9/lstm_68/lstm_cell_68/Sigmoid_1Sigmoid+model_9/lstm_68/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_68/lstm_cell_68/Sigmoid_1Ы
 model_9/lstm_68/lstm_cell_68/mulMul*model_9/lstm_68/lstm_cell_68/Sigmoid_1:y:0 model_9/lstm_68/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_68/lstm_cell_68/mul­
!model_9/lstm_68/lstm_cell_68/ReluRelu+model_9/lstm_68/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2#
!model_9/lstm_68/lstm_cell_68/Reluм
"model_9/lstm_68/lstm_cell_68/mul_1Mul(model_9/lstm_68/lstm_cell_68/Sigmoid:y:0/model_9/lstm_68/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_68/lstm_cell_68/mul_1б
"model_9/lstm_68/lstm_cell_68/add_1AddV2$model_9/lstm_68/lstm_cell_68/mul:z:0&model_9/lstm_68/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_68/lstm_cell_68/add_1К
&model_9/lstm_68/lstm_cell_68/Sigmoid_2Sigmoid+model_9/lstm_68/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_68/lstm_cell_68/Sigmoid_2Ќ
#model_9/lstm_68/lstm_cell_68/Relu_1Relu&model_9/lstm_68/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2%
#model_9/lstm_68/lstm_cell_68/Relu_1р
"model_9/lstm_68/lstm_cell_68/mul_2Mul*model_9/lstm_68/lstm_cell_68/Sigmoid_2:y:01model_9/lstm_68/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_68/lstm_cell_68/mul_2Џ
-model_9/lstm_68/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2/
-model_9/lstm_68/TensorArrayV2_1/element_shapeј
model_9/lstm_68/TensorArrayV2_1TensorListReserve6model_9/lstm_68/TensorArrayV2_1/element_shape:output:0(model_9/lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_9/lstm_68/TensorArrayV2_1n
model_9/lstm_68/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_68/time
(model_9/lstm_68/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model_9/lstm_68/while/maximum_iterations
"model_9/lstm_68/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/lstm_68/while/loop_counterт
model_9/lstm_68/whileWhile+model_9/lstm_68/while/loop_counter:output:01model_9/lstm_68/while/maximum_iterations:output:0model_9/lstm_68/time:output:0(model_9/lstm_68/TensorArrayV2_1:handle:0model_9/lstm_68/zeros:output:0 model_9/lstm_68/zeros_1:output:0(model_9/lstm_68/strided_slice_1:output:0Gmodel_9/lstm_68/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_9_lstm_68_lstm_cell_68_matmul_readvariableop_resource=model_9_lstm_68_lstm_cell_68_matmul_1_readvariableop_resource<model_9_lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#model_9_lstm_68_while_body_50183893*/
cond'R%
#model_9_lstm_68_while_cond_50183892*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
model_9/lstm_68/whileе
@model_9/lstm_68/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2B
@model_9/lstm_68/TensorArrayV2Stack/TensorListStack/element_shapeБ
2model_9/lstm_68/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_68/while:output:3Imodel_9/lstm_68/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype024
2model_9/lstm_68/TensorArrayV2Stack/TensorListStackЁ
%model_9/lstm_68/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%model_9/lstm_68/strided_slice_3/stack
'model_9/lstm_68/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_9/lstm_68/strided_slice_3/stack_1
'model_9/lstm_68/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_68/strided_slice_3/stack_2њ
model_9/lstm_68/strided_slice_3StridedSlice;model_9/lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/lstm_68/strided_slice_3/stack:output:00model_9/lstm_68/strided_slice_3/stack_1:output:00model_9/lstm_68/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2!
model_9/lstm_68/strided_slice_3
 model_9/lstm_68/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_9/lstm_68/transpose_1/permю
model_9/lstm_68/transpose_1	Transpose;model_9/lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0)model_9/lstm_68/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/lstm_68/transpose_1
model_9/lstm_68/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_68/runtimed
model_9/gru_59/ShapeShapeinput_10*
T0*
_output_shapes
:2
model_9/gru_59/Shape
"model_9/gru_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_9/gru_59/strided_slice/stack
$model_9/gru_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/gru_59/strided_slice/stack_1
$model_9/gru_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/gru_59/strided_slice/stack_2М
model_9/gru_59/strided_sliceStridedSlicemodel_9/gru_59/Shape:output:0+model_9/gru_59/strided_slice/stack:output:0-model_9/gru_59/strided_slice/stack_1:output:0-model_9/gru_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/gru_59/strided_slicez
model_9/gru_59/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/gru_59/zeros/mul/yЈ
model_9/gru_59/zeros/mulMul%model_9/gru_59/strided_slice:output:0#model_9/gru_59/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_59/zeros/mul}
model_9/gru_59/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/gru_59/zeros/Less/yЃ
model_9/gru_59/zeros/LessLessmodel_9/gru_59/zeros/mul:z:0$model_9/gru_59/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_59/zeros/Less
model_9/gru_59/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
model_9/gru_59/zeros/packed/1П
model_9/gru_59/zeros/packedPack%model_9/gru_59/strided_slice:output:0&model_9/gru_59/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/gru_59/zeros/packed}
model_9/gru_59/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/gru_59/zeros/ConstБ
model_9/gru_59/zerosFill$model_9/gru_59/zeros/packed:output:0#model_9/gru_59/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_59/zeros
model_9/gru_59/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_9/gru_59/transpose/permВ
model_9/gru_59/transpose	Transposeinput_10&model_9/gru_59/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
model_9/gru_59/transpose|
model_9/gru_59/Shape_1Shapemodel_9/gru_59/transpose:y:0*
T0*
_output_shapes
:2
model_9/gru_59/Shape_1
$model_9/gru_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/gru_59/strided_slice_1/stack
&model_9/gru_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/gru_59/strided_slice_1/stack_1
&model_9/gru_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/gru_59/strided_slice_1/stack_2Ш
model_9/gru_59/strided_slice_1StridedSlicemodel_9/gru_59/Shape_1:output:0-model_9/gru_59/strided_slice_1/stack:output:0/model_9/gru_59/strided_slice_1/stack_1:output:0/model_9/gru_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_9/gru_59/strided_slice_1Ѓ
*model_9/gru_59/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*model_9/gru_59/TensorArrayV2/element_shapeю
model_9/gru_59/TensorArrayV2TensorListReserve3model_9/gru_59/TensorArrayV2/element_shape:output:0'model_9/gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/gru_59/TensorArrayV2н
Dmodel_9/gru_59/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dmodel_9/gru_59/TensorArrayUnstack/TensorListFromTensor/element_shapeД
6model_9/gru_59/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/gru_59/transpose:y:0Mmodel_9/gru_59/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_9/gru_59/TensorArrayUnstack/TensorListFromTensor
$model_9/gru_59/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/gru_59/strided_slice_2/stack
&model_9/gru_59/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/gru_59/strided_slice_2/stack_1
&model_9/gru_59/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/gru_59/strided_slice_2/stack_2ж
model_9/gru_59/strided_slice_2StridedSlicemodel_9/gru_59/transpose:y:0-model_9/gru_59/strided_slice_2/stack:output:0/model_9/gru_59/strided_slice_2/stack_1:output:0/model_9/gru_59/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2 
model_9/gru_59/strided_slice_2Ъ
)model_9/gru_59/gru_cell_59/ReadVariableOpReadVariableOp2model_9_gru_59_gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02+
)model_9/gru_59/gru_cell_59/ReadVariableOpН
"model_9/gru_59/gru_cell_59/unstackUnpack1model_9/gru_59/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2$
"model_9/gru_59/gru_cell_59/unstackп
0model_9/gru_59/gru_cell_59/MatMul/ReadVariableOpReadVariableOp9model_9_gru_59_gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0model_9/gru_59/gru_cell_59/MatMul/ReadVariableOpц
!model_9/gru_59/gru_cell_59/MatMulMatMul'model_9/gru_59/strided_slice_2:output:08model_9/gru_59/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2#
!model_9/gru_59/gru_cell_59/MatMulр
"model_9/gru_59/gru_cell_59/BiasAddBiasAdd+model_9/gru_59/gru_cell_59/MatMul:product:0+model_9/gru_59/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"model_9/gru_59/gru_cell_59/BiasAdd
 model_9/gru_59/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/gru_59/gru_cell_59/ConstЃ
*model_9/gru_59/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*model_9/gru_59/gru_cell_59/split/split_dim
 model_9/gru_59/gru_cell_59/splitSplit3model_9/gru_59/gru_cell_59/split/split_dim:output:0+model_9/gru_59/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 model_9/gru_59/gru_cell_59/splitх
2model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp;model_9_gru_59_gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype024
2model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOpт
#model_9/gru_59/gru_cell_59/MatMul_1MatMulmodel_9/gru_59/zeros:output:0:model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#model_9/gru_59/gru_cell_59/MatMul_1ц
$model_9/gru_59/gru_cell_59/BiasAdd_1BiasAdd-model_9/gru_59/gru_cell_59/MatMul_1:product:0+model_9/gru_59/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2&
$model_9/gru_59/gru_cell_59/BiasAdd_1
"model_9/gru_59/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2$
"model_9/gru_59/gru_cell_59/Const_1Ї
,model_9/gru_59/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,model_9/gru_59/gru_cell_59/split_1/split_dimк
"model_9/gru_59/gru_cell_59/split_1SplitV-model_9/gru_59/gru_cell_59/BiasAdd_1:output:0+model_9/gru_59/gru_cell_59/Const_1:output:05model_9/gru_59/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2$
"model_9/gru_59/gru_cell_59/split_1г
model_9/gru_59/gru_cell_59/addAddV2)model_9/gru_59/gru_cell_59/split:output:0+model_9/gru_59/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_59/gru_cell_59/addЉ
"model_9/gru_59/gru_cell_59/SigmoidSigmoid"model_9/gru_59/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/gru_59/gru_cell_59/Sigmoidз
 model_9/gru_59/gru_cell_59/add_1AddV2)model_9/gru_59/gru_cell_59/split:output:1+model_9/gru_59/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_59/gru_cell_59/add_1Џ
$model_9/gru_59/gru_cell_59/Sigmoid_1Sigmoid$model_9/gru_59/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_59/gru_cell_59/Sigmoid_1а
model_9/gru_59/gru_cell_59/mulMul(model_9/gru_59/gru_cell_59/Sigmoid_1:y:0+model_9/gru_59/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_59/gru_cell_59/mulЮ
 model_9/gru_59/gru_cell_59/add_2AddV2)model_9/gru_59/gru_cell_59/split:output:2"model_9/gru_59/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_59/gru_cell_59/add_2Ђ
model_9/gru_59/gru_cell_59/ReluRelu$model_9/gru_59/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
model_9/gru_59/gru_cell_59/ReluФ
 model_9/gru_59/gru_cell_59/mul_1Mul&model_9/gru_59/gru_cell_59/Sigmoid:y:0model_9/gru_59/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_59/gru_cell_59/mul_1
 model_9/gru_59/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 model_9/gru_59/gru_cell_59/sub/xЬ
model_9/gru_59/gru_cell_59/subSub)model_9/gru_59/gru_cell_59/sub/x:output:0&model_9/gru_59/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_59/gru_cell_59/subа
 model_9/gru_59/gru_cell_59/mul_2Mul"model_9/gru_59/gru_cell_59/sub:z:0-model_9/gru_59/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_59/gru_cell_59/mul_2Ы
 model_9/gru_59/gru_cell_59/add_3AddV2$model_9/gru_59/gru_cell_59/mul_1:z:0$model_9/gru_59/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_59/gru_cell_59/add_3­
,model_9/gru_59/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2.
,model_9/gru_59/TensorArrayV2_1/element_shapeє
model_9/gru_59/TensorArrayV2_1TensorListReserve5model_9/gru_59/TensorArrayV2_1/element_shape:output:0'model_9/gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_9/gru_59/TensorArrayV2_1l
model_9/gru_59/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/gru_59/time
'model_9/gru_59/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'model_9/gru_59/while/maximum_iterations
!model_9/gru_59/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_9/gru_59/while/loop_counter
model_9/gru_59/whileWhile*model_9/gru_59/while/loop_counter:output:00model_9/gru_59/while/maximum_iterations:output:0model_9/gru_59/time:output:0'model_9/gru_59/TensorArrayV2_1:handle:0model_9/gru_59/zeros:output:0'model_9/gru_59/strided_slice_1:output:0Fmodel_9/gru_59/TensorArrayUnstack/TensorListFromTensor:output_handle:02model_9_gru_59_gru_cell_59_readvariableop_resource9model_9_gru_59_gru_cell_59_matmul_readvariableop_resource;model_9_gru_59_gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*.
body&R$
"model_9_gru_59_while_body_50184043*.
cond&R$
"model_9_gru_59_while_cond_50184042*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
model_9/gru_59/whileг
?model_9/gru_59/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2A
?model_9/gru_59/TensorArrayV2Stack/TensorListStack/element_shape­
1model_9/gru_59/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/gru_59/while:output:3Hmodel_9/gru_59/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype023
1model_9/gru_59/TensorArrayV2Stack/TensorListStack
$model_9/gru_59/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2&
$model_9/gru_59/strided_slice_3/stack
&model_9/gru_59/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/gru_59/strided_slice_3/stack_1
&model_9/gru_59/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/gru_59/strided_slice_3/stack_2є
model_9/gru_59/strided_slice_3StridedSlice:model_9/gru_59/TensorArrayV2Stack/TensorListStack:tensor:0-model_9/gru_59/strided_slice_3/stack:output:0/model_9/gru_59/strided_slice_3/stack_1:output:0/model_9/gru_59/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2 
model_9/gru_59/strided_slice_3
model_9/gru_59/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_9/gru_59/transpose_1/permъ
model_9/gru_59/transpose_1	Transpose:model_9/gru_59/TensorArrayV2Stack/TensorListStack:tensor:0(model_9/gru_59/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
model_9/gru_59/transpose_1
model_9/gru_59/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/gru_59/runtimeІ
model_9/dropout_68/IdentityIdentitymodel_9/lstm_68/transpose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/dropout_68/IdentityЁ
model_9/dropout_69/IdentityIdentity'model_9/gru_59/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/dropout_69/Identity
model_9/lstm_69/ShapeShape$model_9/dropout_68/Identity:output:0*
T0*
_output_shapes
:2
model_9/lstm_69/Shape
#model_9/lstm_69/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/lstm_69/strided_slice/stack
%model_9/lstm_69/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_69/strided_slice/stack_1
%model_9/lstm_69/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_69/strided_slice/stack_2Т
model_9/lstm_69/strided_sliceStridedSlicemodel_9/lstm_69/Shape:output:0,model_9/lstm_69/strided_slice/stack:output:0.model_9/lstm_69/strided_slice/stack_1:output:0.model_9/lstm_69/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_69/strided_slice|
model_9/lstm_69/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/lstm_69/zeros/mul/yЌ
model_9/lstm_69/zeros/mulMul&model_9/lstm_69/strided_slice:output:0$model_9/lstm_69/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/zeros/mul
model_9/lstm_69/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/lstm_69/zeros/Less/yЇ
model_9/lstm_69/zeros/LessLessmodel_9/lstm_69/zeros/mul:z:0%model_9/lstm_69/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/zeros/Less
model_9/lstm_69/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
model_9/lstm_69/zeros/packed/1У
model_9/lstm_69/zeros/packedPack&model_9/lstm_69/strided_slice:output:0'model_9/lstm_69/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_69/zeros/packed
model_9/lstm_69/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_69/zeros/ConstЕ
model_9/lstm_69/zerosFill%model_9/lstm_69/zeros/packed:output:0$model_9/lstm_69/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/lstm_69/zeros
model_9/lstm_69/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/lstm_69/zeros_1/mul/yВ
model_9/lstm_69/zeros_1/mulMul&model_9/lstm_69/strided_slice:output:0&model_9/lstm_69/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/zeros_1/mul
model_9/lstm_69/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
model_9/lstm_69/zeros_1/Less/yЏ
model_9/lstm_69/zeros_1/LessLessmodel_9/lstm_69/zeros_1/mul:z:0'model_9/lstm_69/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/zeros_1/Less
 model_9/lstm_69/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 model_9/lstm_69/zeros_1/packed/1Щ
model_9/lstm_69/zeros_1/packedPack&model_9/lstm_69/strided_slice:output:0)model_9/lstm_69/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_9/lstm_69/zeros_1/packed
model_9/lstm_69/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_69/zeros_1/ConstН
model_9/lstm_69/zeros_1Fill'model_9/lstm_69/zeros_1/packed:output:0&model_9/lstm_69/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/lstm_69/zeros_1
model_9/lstm_69/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_9/lstm_69/transpose/permб
model_9/lstm_69/transpose	Transpose$model_9/dropout_68/Identity:output:0'model_9/lstm_69/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/lstm_69/transpose
model_9/lstm_69/Shape_1Shapemodel_9/lstm_69/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_69/Shape_1
%model_9/lstm_69/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_69/strided_slice_1/stack
'model_9/lstm_69/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_69/strided_slice_1/stack_1
'model_9/lstm_69/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_69/strided_slice_1/stack_2Ю
model_9/lstm_69/strided_slice_1StridedSlice model_9/lstm_69/Shape_1:output:0.model_9/lstm_69/strided_slice_1/stack:output:00model_9/lstm_69/strided_slice_1/stack_1:output:00model_9/lstm_69/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_9/lstm_69/strided_slice_1Ѕ
+model_9/lstm_69/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+model_9/lstm_69/TensorArrayV2/element_shapeђ
model_9/lstm_69/TensorArrayV2TensorListReserve4model_9/lstm_69/TensorArrayV2/element_shape:output:0(model_9/lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_69/TensorArrayV2п
Emodel_9/lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2G
Emodel_9/lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7model_9/lstm_69/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_69/transpose:y:0Nmodel_9/lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_9/lstm_69/TensorArrayUnstack/TensorListFromTensor
%model_9/lstm_69/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_69/strided_slice_2/stack
'model_9/lstm_69/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_69/strided_slice_2/stack_1
'model_9/lstm_69/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_69/strided_slice_2/stack_2м
model_9/lstm_69/strided_slice_2StridedSlicemodel_9/lstm_69/transpose:y:0.model_9/lstm_69/strided_slice_2/stack:output:00model_9/lstm_69/strided_slice_2/stack_1:output:00model_9/lstm_69/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2!
model_9/lstm_69/strided_slice_2х
2model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp;model_9_lstm_69_lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype024
2model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOpэ
#model_9/lstm_69/lstm_cell_69/MatMulMatMul(model_9/lstm_69/strided_slice_2:output:0:model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#model_9/lstm_69/lstm_cell_69/MatMulы
4model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp=model_9_lstm_69_lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype026
4model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpщ
%model_9/lstm_69/lstm_cell_69/MatMul_1MatMulmodel_9/lstm_69/zeros:output:0<model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%model_9/lstm_69/lstm_cell_69/MatMul_1р
 model_9/lstm_69/lstm_cell_69/addAddV2-model_9/lstm_69/lstm_cell_69/MatMul:product:0/model_9/lstm_69/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2"
 model_9/lstm_69/lstm_cell_69/addф
3model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp<model_9_lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype025
3model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpэ
$model_9/lstm_69/lstm_cell_69/BiasAddBiasAdd$model_9/lstm_69/lstm_cell_69/add:z:0;model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$model_9/lstm_69/lstm_cell_69/BiasAdd
"model_9/lstm_69/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/lstm_69/lstm_cell_69/Const
,model_9/lstm_69/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_9/lstm_69/lstm_cell_69/split/split_dimГ
"model_9/lstm_69/lstm_cell_69/splitSplit5model_9/lstm_69/lstm_cell_69/split/split_dim:output:0-model_9/lstm_69/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2$
"model_9/lstm_69/lstm_cell_69/splitЖ
$model_9/lstm_69/lstm_cell_69/SigmoidSigmoid+model_9/lstm_69/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/lstm_69/lstm_cell_69/SigmoidК
&model_9/lstm_69/lstm_cell_69/Sigmoid_1Sigmoid+model_9/lstm_69/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_69/lstm_cell_69/Sigmoid_1Ы
 model_9/lstm_69/lstm_cell_69/mulMul*model_9/lstm_69/lstm_cell_69/Sigmoid_1:y:0 model_9/lstm_69/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_69/lstm_cell_69/mul­
!model_9/lstm_69/lstm_cell_69/ReluRelu+model_9/lstm_69/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22#
!model_9/lstm_69/lstm_cell_69/Reluм
"model_9/lstm_69/lstm_cell_69/mul_1Mul(model_9/lstm_69/lstm_cell_69/Sigmoid:y:0/model_9/lstm_69/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_69/lstm_cell_69/mul_1б
"model_9/lstm_69/lstm_cell_69/add_1AddV2$model_9/lstm_69/lstm_cell_69/mul:z:0&model_9/lstm_69/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_69/lstm_cell_69/add_1К
&model_9/lstm_69/lstm_cell_69/Sigmoid_2Sigmoid+model_9/lstm_69/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_69/lstm_cell_69/Sigmoid_2Ќ
#model_9/lstm_69/lstm_cell_69/Relu_1Relu&model_9/lstm_69/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#model_9/lstm_69/lstm_cell_69/Relu_1р
"model_9/lstm_69/lstm_cell_69/mul_2Mul*model_9/lstm_69/lstm_cell_69/Sigmoid_2:y:01model_9/lstm_69/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_69/lstm_cell_69/mul_2Џ
-model_9/lstm_69/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-model_9/lstm_69/TensorArrayV2_1/element_shapeј
model_9/lstm_69/TensorArrayV2_1TensorListReserve6model_9/lstm_69/TensorArrayV2_1/element_shape:output:0(model_9/lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_9/lstm_69/TensorArrayV2_1n
model_9/lstm_69/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_69/time
(model_9/lstm_69/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model_9/lstm_69/while/maximum_iterations
"model_9/lstm_69/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/lstm_69/while/loop_counterт
model_9/lstm_69/whileWhile+model_9/lstm_69/while/loop_counter:output:01model_9/lstm_69/while/maximum_iterations:output:0model_9/lstm_69/time:output:0(model_9/lstm_69/TensorArrayV2_1:handle:0model_9/lstm_69/zeros:output:0 model_9/lstm_69/zeros_1:output:0(model_9/lstm_69/strided_slice_1:output:0Gmodel_9/lstm_69/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_9_lstm_69_lstm_cell_69_matmul_readvariableop_resource=model_9_lstm_69_lstm_cell_69_matmul_1_readvariableop_resource<model_9_lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#model_9_lstm_69_while_body_50184199*/
cond'R%
#model_9_lstm_69_while_cond_50184198*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
model_9/lstm_69/whileе
@model_9/lstm_69/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@model_9/lstm_69/TensorArrayV2Stack/TensorListStack/element_shapeБ
2model_9/lstm_69/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_69/while:output:3Imodel_9/lstm_69/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2model_9/lstm_69/TensorArrayV2Stack/TensorListStackЁ
%model_9/lstm_69/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%model_9/lstm_69/strided_slice_3/stack
'model_9/lstm_69/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_9/lstm_69/strided_slice_3/stack_1
'model_9/lstm_69/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_69/strided_slice_3/stack_2њ
model_9/lstm_69/strided_slice_3StridedSlice;model_9/lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/lstm_69/strided_slice_3/stack:output:00model_9/lstm_69/strided_slice_3/stack_1:output:00model_9/lstm_69/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
model_9/lstm_69/strided_slice_3
 model_9/lstm_69/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_9/lstm_69/transpose_1/permю
model_9/lstm_69/transpose_1	Transpose;model_9/lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0)model_9/lstm_69/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
model_9/lstm_69/transpose_1
model_9/lstm_69/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_69/runtimeУ
'model_9/dense_327/MatMul/ReadVariableOpReadVariableOp0model_9_dense_327_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02)
'model_9/dense_327/MatMul/ReadVariableOpЫ
model_9/dense_327/MatMulMatMul(model_9/lstm_69/strided_slice_3:output:0/model_9/dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_327/MatMulТ
(model_9/dense_327/BiasAdd/ReadVariableOpReadVariableOp1model_9_dense_327_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_9/dense_327/BiasAdd/ReadVariableOpЩ
model_9/dense_327/BiasAddBiasAdd"model_9/dense_327/MatMul:product:00model_9/dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_327/BiasAdd
model_9/dense_327/ReluRelu"model_9/dense_327/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_327/ReluУ
'model_9/dense_328/MatMul/ReadVariableOpReadVariableOp0model_9_dense_328_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02)
'model_9/dense_328/MatMul/ReadVariableOpЧ
model_9/dense_328/MatMulMatMul$model_9/dropout_69/Identity:output:0/model_9/dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_328/MatMulТ
(model_9/dense_328/BiasAdd/ReadVariableOpReadVariableOp1model_9_dense_328_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_9/dense_328/BiasAdd/ReadVariableOpЩ
model_9/dense_328/BiasAddBiasAdd"model_9/dense_328/MatMul:product:00model_9/dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_328/BiasAdd
model_9/dense_328/ReluRelu"model_9/dense_328/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_328/Relu
!model_9/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_9/concatenate_9/concat/axisћ
model_9/concatenate_9/concatConcatV2$model_9/dense_327/Relu:activations:0$model_9/dense_328/Relu:activations:0*model_9/concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
model_9/concatenate_9/concatУ
'model_9/dense_329/MatMul/ReadVariableOpReadVariableOp0model_9_dense_329_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02)
'model_9/dense_329/MatMul/ReadVariableOpШ
model_9/dense_329/MatMulMatMul%model_9/concatenate_9/concat:output:0/model_9/dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_9/dense_329/MatMulТ
(model_9/dense_329/BiasAdd/ReadVariableOpReadVariableOp1model_9_dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_9/dense_329/BiasAdd/ReadVariableOpЩ
model_9/dense_329/BiasAddBiasAdd"model_9/dense_329/MatMul:product:00model_9/dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_9/dense_329/BiasAdd
IdentityIdentity"model_9/dense_329/BiasAdd:output:0)^model_9/dense_327/BiasAdd/ReadVariableOp(^model_9/dense_327/MatMul/ReadVariableOp)^model_9/dense_328/BiasAdd/ReadVariableOp(^model_9/dense_328/MatMul/ReadVariableOp)^model_9/dense_329/BiasAdd/ReadVariableOp(^model_9/dense_329/MatMul/ReadVariableOp1^model_9/gru_59/gru_cell_59/MatMul/ReadVariableOp3^model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOp*^model_9/gru_59/gru_cell_59/ReadVariableOp^model_9/gru_59/while4^model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp3^model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOp5^model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp^model_9/lstm_68/while4^model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp3^model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOp5^model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp^model_9/lstm_69/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2T
(model_9/dense_327/BiasAdd/ReadVariableOp(model_9/dense_327/BiasAdd/ReadVariableOp2R
'model_9/dense_327/MatMul/ReadVariableOp'model_9/dense_327/MatMul/ReadVariableOp2T
(model_9/dense_328/BiasAdd/ReadVariableOp(model_9/dense_328/BiasAdd/ReadVariableOp2R
'model_9/dense_328/MatMul/ReadVariableOp'model_9/dense_328/MatMul/ReadVariableOp2T
(model_9/dense_329/BiasAdd/ReadVariableOp(model_9/dense_329/BiasAdd/ReadVariableOp2R
'model_9/dense_329/MatMul/ReadVariableOp'model_9/dense_329/MatMul/ReadVariableOp2d
0model_9/gru_59/gru_cell_59/MatMul/ReadVariableOp0model_9/gru_59/gru_cell_59/MatMul/ReadVariableOp2h
2model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOp2model_9/gru_59/gru_cell_59/MatMul_1/ReadVariableOp2V
)model_9/gru_59/gru_cell_59/ReadVariableOp)model_9/gru_59/gru_cell_59/ReadVariableOp2,
model_9/gru_59/whilemodel_9/gru_59/while2j
3model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp3model_9/lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp2h
2model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOp2model_9/lstm_68/lstm_cell_68/MatMul/ReadVariableOp2l
4model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp4model_9/lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp2.
model_9/lstm_68/whilemodel_9/lstm_68/while2j
3model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp3model_9/lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp2h
2model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOp2model_9/lstm_69/lstm_cell_69/MatMul/ReadVariableOp2l
4model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp4model_9/lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp2.
model_9/lstm_69/whilemodel_9/lstm_69/while:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
`
ђ

"model_9_gru_59_while_body_50184043:
6model_9_gru_59_while_model_9_gru_59_while_loop_counter@
<model_9_gru_59_while_model_9_gru_59_while_maximum_iterations$
 model_9_gru_59_while_placeholder&
"model_9_gru_59_while_placeholder_1&
"model_9_gru_59_while_placeholder_29
5model_9_gru_59_while_model_9_gru_59_strided_slice_1_0u
qmodel_9_gru_59_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_59_tensorarrayunstack_tensorlistfromtensor_0>
:model_9_gru_59_while_gru_cell_59_readvariableop_resource_0E
Amodel_9_gru_59_while_gru_cell_59_matmul_readvariableop_resource_0G
Cmodel_9_gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0!
model_9_gru_59_while_identity#
model_9_gru_59_while_identity_1#
model_9_gru_59_while_identity_2#
model_9_gru_59_while_identity_3#
model_9_gru_59_while_identity_47
3model_9_gru_59_while_model_9_gru_59_strided_slice_1s
omodel_9_gru_59_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_59_tensorarrayunstack_tensorlistfromtensor<
8model_9_gru_59_while_gru_cell_59_readvariableop_resourceC
?model_9_gru_59_while_gru_cell_59_matmul_readvariableop_resourceE
Amodel_9_gru_59_while_gru_cell_59_matmul_1_readvariableop_resourceЂ6model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOpЂ8model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpЂ/model_9/gru_59/while/gru_cell_59/ReadVariableOpс
Fmodel_9/gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
Fmodel_9/gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8model_9/gru_59/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_9_gru_59_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_59_tensorarrayunstack_tensorlistfromtensor_0 model_9_gru_59_while_placeholderOmodel_9/gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02:
8model_9/gru_59/while/TensorArrayV2Read/TensorListGetItemо
/model_9/gru_59/while/gru_cell_59/ReadVariableOpReadVariableOp:model_9_gru_59_while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/model_9/gru_59/while/gru_cell_59/ReadVariableOpЯ
(model_9/gru_59/while/gru_cell_59/unstackUnpack7model_9/gru_59/while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2*
(model_9/gru_59/while/gru_cell_59/unstackѓ
6model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOpReadVariableOpAmodel_9_gru_59_while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype028
6model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp
'model_9/gru_59/while/gru_cell_59/MatMulMatMul?model_9/gru_59/while/TensorArrayV2Read/TensorListGetItem:item:0>model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2)
'model_9/gru_59/while/gru_cell_59/MatMulј
(model_9/gru_59/while/gru_cell_59/BiasAddBiasAdd1model_9/gru_59/while/gru_cell_59/MatMul:product:01model_9/gru_59/while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2*
(model_9/gru_59/while/gru_cell_59/BiasAdd
&model_9/gru_59/while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_9/gru_59/while/gru_cell_59/ConstЏ
0model_9/gru_59/while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ22
0model_9/gru_59/while/gru_cell_59/split/split_dimА
&model_9/gru_59/while/gru_cell_59/splitSplit9model_9/gru_59/while/gru_cell_59/split/split_dim:output:01model_9/gru_59/while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&model_9/gru_59/while/gru_cell_59/splitљ
8model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOpCmodel_9_gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02:
8model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpљ
)model_9/gru_59/while/gru_cell_59/MatMul_1MatMul"model_9_gru_59_while_placeholder_2@model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2+
)model_9/gru_59/while/gru_cell_59/MatMul_1ў
*model_9/gru_59/while/gru_cell_59/BiasAdd_1BiasAdd3model_9/gru_59/while/gru_cell_59/MatMul_1:product:01model_9/gru_59/while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2,
*model_9/gru_59/while/gru_cell_59/BiasAdd_1Љ
(model_9/gru_59/while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2*
(model_9/gru_59/while/gru_cell_59/Const_1Г
2model_9/gru_59/while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2model_9/gru_59/while/gru_cell_59/split_1/split_dimј
(model_9/gru_59/while/gru_cell_59/split_1SplitV3model_9/gru_59/while/gru_cell_59/BiasAdd_1:output:01model_9/gru_59/while/gru_cell_59/Const_1:output:0;model_9/gru_59/while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2*
(model_9/gru_59/while/gru_cell_59/split_1ы
$model_9/gru_59/while/gru_cell_59/addAddV2/model_9/gru_59/while/gru_cell_59/split:output:01model_9/gru_59/while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_59/while/gru_cell_59/addЛ
(model_9/gru_59/while/gru_cell_59/SigmoidSigmoid(model_9/gru_59/while/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/gru_59/while/gru_cell_59/Sigmoidя
&model_9/gru_59/while/gru_cell_59/add_1AddV2/model_9/gru_59/while/gru_cell_59/split:output:11model_9/gru_59/while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_59/while/gru_cell_59/add_1С
*model_9/gru_59/while/gru_cell_59/Sigmoid_1Sigmoid*model_9/gru_59/while/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*model_9/gru_59/while/gru_cell_59/Sigmoid_1ш
$model_9/gru_59/while/gru_cell_59/mulMul.model_9/gru_59/while/gru_cell_59/Sigmoid_1:y:01model_9/gru_59/while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_59/while/gru_cell_59/mulц
&model_9/gru_59/while/gru_cell_59/add_2AddV2/model_9/gru_59/while/gru_cell_59/split:output:2(model_9/gru_59/while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_59/while/gru_cell_59/add_2Д
%model_9/gru_59/while/gru_cell_59/ReluRelu*model_9/gru_59/while/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22'
%model_9/gru_59/while/gru_cell_59/Reluл
&model_9/gru_59/while/gru_cell_59/mul_1Mul,model_9/gru_59/while/gru_cell_59/Sigmoid:y:0"model_9_gru_59_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_59/while/gru_cell_59/mul_1
&model_9/gru_59/while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&model_9/gru_59/while/gru_cell_59/sub/xф
$model_9/gru_59/while/gru_cell_59/subSub/model_9/gru_59/while/gru_cell_59/sub/x:output:0,model_9/gru_59/while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_59/while/gru_cell_59/subш
&model_9/gru_59/while/gru_cell_59/mul_2Mul(model_9/gru_59/while/gru_cell_59/sub:z:03model_9/gru_59/while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_59/while/gru_cell_59/mul_2у
&model_9/gru_59/while/gru_cell_59/add_3AddV2*model_9/gru_59/while/gru_cell_59/mul_1:z:0*model_9/gru_59/while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_59/while/gru_cell_59/add_3Њ
9model_9/gru_59/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_9_gru_59_while_placeholder_1 model_9_gru_59_while_placeholder*model_9/gru_59/while/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02;
9model_9/gru_59/while/TensorArrayV2Write/TensorListSetItemz
model_9/gru_59/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/gru_59/while/add/yЅ
model_9/gru_59/while/addAddV2 model_9_gru_59_while_placeholder#model_9/gru_59/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_59/while/add~
model_9/gru_59/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/gru_59/while/add_1/yС
model_9/gru_59/while/add_1AddV26model_9_gru_59_while_model_9_gru_59_while_loop_counter%model_9/gru_59/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_59/while/add_1Б
model_9/gru_59/while/IdentityIdentitymodel_9/gru_59/while/add_1:z:07^model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp9^model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0^model_9/gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
model_9/gru_59/while/Identityг
model_9/gru_59/while/Identity_1Identity<model_9_gru_59_while_model_9_gru_59_while_maximum_iterations7^model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp9^model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0^model_9/gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/gru_59/while/Identity_1Г
model_9/gru_59/while/Identity_2Identitymodel_9/gru_59/while/add:z:07^model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp9^model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0^model_9/gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/gru_59/while/Identity_2р
model_9/gru_59/while/Identity_3IdentityImodel_9/gru_59/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp9^model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0^model_9/gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/gru_59/while/Identity_3в
model_9/gru_59/while/Identity_4Identity*model_9/gru_59/while/gru_cell_59/add_3:z:07^model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp9^model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0^model_9/gru_59/while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22!
model_9/gru_59/while/Identity_4"
Amodel_9_gru_59_while_gru_cell_59_matmul_1_readvariableop_resourceCmodel_9_gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0"
?model_9_gru_59_while_gru_cell_59_matmul_readvariableop_resourceAmodel_9_gru_59_while_gru_cell_59_matmul_readvariableop_resource_0"v
8model_9_gru_59_while_gru_cell_59_readvariableop_resource:model_9_gru_59_while_gru_cell_59_readvariableop_resource_0"G
model_9_gru_59_while_identity&model_9/gru_59/while/Identity:output:0"K
model_9_gru_59_while_identity_1(model_9/gru_59/while/Identity_1:output:0"K
model_9_gru_59_while_identity_2(model_9/gru_59/while/Identity_2:output:0"K
model_9_gru_59_while_identity_3(model_9/gru_59/while/Identity_3:output:0"K
model_9_gru_59_while_identity_4(model_9/gru_59/while/Identity_4:output:0"l
3model_9_gru_59_while_model_9_gru_59_strided_slice_15model_9_gru_59_while_model_9_gru_59_strided_slice_1_0"ф
omodel_9_gru_59_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_59_tensorarrayunstack_tensorlistfromtensorqmodel_9_gru_59_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_59_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2p
6model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp6model_9/gru_59/while/gru_cell_59/MatMul/ReadVariableOp2t
8model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp8model_9/gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp2b
/model_9/gru_59/while/gru_cell_59/ReadVariableOp/model_9/gru_59/while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ж

*__inference_lstm_68_layer_call_fn_50189202

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501862442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 

)__inference_gru_59_layer_call_fn_50189920
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501854692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0


Ф
&__inference_signature_wrapper_50187511
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_501843062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
њG
А
while_body_50189808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
З
u
K__inference_concatenate_9_layer_call_and_return_conditional_losses_50187230

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
C

while_body_50186312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_50189807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50189807___redundant_placeholder06
2while_while_cond_50189807___redundant_placeholder16
2while_while_cond_50189807___redundant_placeholder26
2while_while_cond_50189807___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ы
f
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190593

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
І

э
lstm_68_while_cond_50187578,
(lstm_68_while_lstm_68_while_loop_counter2
.lstm_68_while_lstm_68_while_maximum_iterations
lstm_68_while_placeholder
lstm_68_while_placeholder_1
lstm_68_while_placeholder_2
lstm_68_while_placeholder_3.
*lstm_68_while_less_lstm_68_strided_slice_1F
Blstm_68_while_lstm_68_while_cond_50187578___redundant_placeholder0F
Blstm_68_while_lstm_68_while_cond_50187578___redundant_placeholder1F
Blstm_68_while_lstm_68_while_cond_50187578___redundant_placeholder2F
Blstm_68_while_lstm_68_while_cond_50187578___redundant_placeholder3
lstm_68_while_identity

lstm_68/while/LessLesslstm_68_while_placeholder*lstm_68_while_less_lstm_68_strided_slice_1*
T0*
_output_shapes
: 2
lstm_68/while/Lessu
lstm_68/while/IdentityIdentitylstm_68/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_68/while/Identity"9
lstm_68_while_identitylstm_68/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
ди

E__inference_model_9_layer_call_and_return_conditional_losses_50188487

inputs7
3lstm_68_lstm_cell_68_matmul_readvariableop_resource9
5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource8
4lstm_68_lstm_cell_68_biasadd_readvariableop_resource.
*gru_59_gru_cell_59_readvariableop_resource5
1gru_59_gru_cell_59_matmul_readvariableop_resource7
3gru_59_gru_cell_59_matmul_1_readvariableop_resource7
3lstm_69_lstm_cell_69_matmul_readvariableop_resource9
5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource8
4lstm_69_lstm_cell_69_biasadd_readvariableop_resource,
(dense_327_matmul_readvariableop_resource-
)dense_327_biasadd_readvariableop_resource,
(dense_328_matmul_readvariableop_resource-
)dense_328_biasadd_readvariableop_resource,
(dense_329_matmul_readvariableop_resource-
)dense_329_biasadd_readvariableop_resource
identityЂ dense_327/BiasAdd/ReadVariableOpЂdense_327/MatMul/ReadVariableOpЂ dense_328/BiasAdd/ReadVariableOpЂdense_328/MatMul/ReadVariableOpЂ dense_329/BiasAdd/ReadVariableOpЂdense_329/MatMul/ReadVariableOpЂ(gru_59/gru_cell_59/MatMul/ReadVariableOpЂ*gru_59/gru_cell_59/MatMul_1/ReadVariableOpЂ!gru_59/gru_cell_59/ReadVariableOpЂgru_59/whileЂ+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpЂ*lstm_68/lstm_cell_68/MatMul/ReadVariableOpЂ,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpЂlstm_68/whileЂ+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpЂ*lstm_69/lstm_cell_69/MatMul/ReadVariableOpЂ,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpЂlstm_69/whileT
lstm_68/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_68/Shape
lstm_68/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice/stack
lstm_68/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_68/strided_slice/stack_1
lstm_68/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_68/strided_slice/stack_2
lstm_68/strided_sliceStridedSlicelstm_68/Shape:output:0$lstm_68/strided_slice/stack:output:0&lstm_68/strided_slice/stack_1:output:0&lstm_68/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_68/strided_slicel
lstm_68/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros/mul/y
lstm_68/zeros/mulMullstm_68/strided_slice:output:0lstm_68/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros/mulo
lstm_68/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_68/zeros/Less/y
lstm_68/zeros/LessLesslstm_68/zeros/mul:z:0lstm_68/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros/Lessr
lstm_68/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros/packed/1Ѓ
lstm_68/zeros/packedPacklstm_68/strided_slice:output:0lstm_68/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_68/zeros/packedo
lstm_68/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/zeros/Const
lstm_68/zerosFilllstm_68/zeros/packed:output:0lstm_68/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/zerosp
lstm_68/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros_1/mul/y
lstm_68/zeros_1/mulMullstm_68/strided_slice:output:0lstm_68/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros_1/muls
lstm_68/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_68/zeros_1/Less/y
lstm_68/zeros_1/LessLesslstm_68/zeros_1/mul:z:0lstm_68/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros_1/Lessv
lstm_68/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros_1/packed/1Љ
lstm_68/zeros_1/packedPacklstm_68/strided_slice:output:0!lstm_68/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_68/zeros_1/packeds
lstm_68/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/zeros_1/Const
lstm_68/zeros_1Filllstm_68/zeros_1/packed:output:0lstm_68/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/zeros_1
lstm_68/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_68/transpose/perm
lstm_68/transpose	Transposeinputslstm_68/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
lstm_68/transposeg
lstm_68/Shape_1Shapelstm_68/transpose:y:0*
T0*
_output_shapes
:2
lstm_68/Shape_1
lstm_68/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice_1/stack
lstm_68/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_1/stack_1
lstm_68/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_1/stack_2
lstm_68/strided_slice_1StridedSlicelstm_68/Shape_1:output:0&lstm_68/strided_slice_1/stack:output:0(lstm_68/strided_slice_1/stack_1:output:0(lstm_68/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_68/strided_slice_1
#lstm_68/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_68/TensorArrayV2/element_shapeв
lstm_68/TensorArrayV2TensorListReserve,lstm_68/TensorArrayV2/element_shape:output:0 lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_68/TensorArrayV2Я
=lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_68/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_68/transpose:y:0Flstm_68/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_68/TensorArrayUnstack/TensorListFromTensor
lstm_68/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice_2/stack
lstm_68/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_2/stack_1
lstm_68/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_2/stack_2Ќ
lstm_68/strided_slice_2StridedSlicelstm_68/transpose:y:0&lstm_68/strided_slice_2/stack:output:0(lstm_68/strided_slice_2/stack_1:output:0(lstm_68/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_68/strided_slice_2Э
*lstm_68/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3lstm_68_lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02,
*lstm_68/lstm_cell_68/MatMul/ReadVariableOpЭ
lstm_68/lstm_cell_68/MatMulMatMul lstm_68/strided_slice_2:output:02lstm_68/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/MatMulг
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02.
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpЩ
lstm_68/lstm_cell_68/MatMul_1MatMullstm_68/zeros:output:04lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/MatMul_1Р
lstm_68/lstm_cell_68/addAddV2%lstm_68/lstm_cell_68/MatMul:product:0'lstm_68/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/addЬ
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02-
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpЭ
lstm_68/lstm_cell_68/BiasAddBiasAddlstm_68/lstm_cell_68/add:z:03lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/BiasAddz
lstm_68/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/lstm_cell_68/Const
$lstm_68/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_68/lstm_cell_68/split/split_dim
lstm_68/lstm_cell_68/splitSplit-lstm_68/lstm_cell_68/split/split_dim:output:0%lstm_68/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_68/lstm_cell_68/split
lstm_68/lstm_cell_68/SigmoidSigmoid#lstm_68/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/SigmoidЂ
lstm_68/lstm_cell_68/Sigmoid_1Sigmoid#lstm_68/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/lstm_cell_68/Sigmoid_1Ћ
lstm_68/lstm_cell_68/mulMul"lstm_68/lstm_cell_68/Sigmoid_1:y:0lstm_68/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul
lstm_68/lstm_cell_68/ReluRelu#lstm_68/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/ReluМ
lstm_68/lstm_cell_68/mul_1Mul lstm_68/lstm_cell_68/Sigmoid:y:0'lstm_68/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul_1Б
lstm_68/lstm_cell_68/add_1AddV2lstm_68/lstm_cell_68/mul:z:0lstm_68/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/add_1Ђ
lstm_68/lstm_cell_68/Sigmoid_2Sigmoid#lstm_68/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/lstm_cell_68/Sigmoid_2
lstm_68/lstm_cell_68/Relu_1Relulstm_68/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/Relu_1Р
lstm_68/lstm_cell_68/mul_2Mul"lstm_68/lstm_cell_68/Sigmoid_2:y:0)lstm_68/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul_2
%lstm_68/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2'
%lstm_68/TensorArrayV2_1/element_shapeи
lstm_68/TensorArrayV2_1TensorListReserve.lstm_68/TensorArrayV2_1/element_shape:output:0 lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_68/TensorArrayV2_1^
lstm_68/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_68/time
 lstm_68/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_68/while/maximum_iterationsz
lstm_68/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_68/while/loop_counterъ
lstm_68/whileWhile#lstm_68/while/loop_counter:output:0)lstm_68/while/maximum_iterations:output:0lstm_68/time:output:0 lstm_68/TensorArrayV2_1:handle:0lstm_68/zeros:output:0lstm_68/zeros_1:output:0 lstm_68/strided_slice_1:output:0?lstm_68/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_68_lstm_cell_68_matmul_readvariableop_resource5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource4lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_68_while_body_50188074*'
condR
lstm_68_while_cond_50188073*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
lstm_68/whileХ
8lstm_68/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2:
8lstm_68/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_68/TensorArrayV2Stack/TensorListStackTensorListStacklstm_68/while:output:3Alstm_68/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02,
*lstm_68/TensorArrayV2Stack/TensorListStack
lstm_68/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_68/strided_slice_3/stack
lstm_68/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_68/strided_slice_3/stack_1
lstm_68/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_3/stack_2Ъ
lstm_68/strided_slice_3StridedSlice3lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_68/strided_slice_3/stack:output:0(lstm_68/strided_slice_3/stack_1:output:0(lstm_68/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_68/strided_slice_3
lstm_68/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_68/transpose_1/permЮ
lstm_68/transpose_1	Transpose3lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_68/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_68/transpose_1v
lstm_68/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/runtimeR
gru_59/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_59/Shape
gru_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice/stack
gru_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_59/strided_slice/stack_1
gru_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_59/strided_slice/stack_2
gru_59/strided_sliceStridedSlicegru_59/Shape:output:0#gru_59/strided_slice/stack:output:0%gru_59/strided_slice/stack_1:output:0%gru_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_59/strided_slicej
gru_59/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_59/zeros/mul/y
gru_59/zeros/mulMulgru_59/strided_slice:output:0gru_59/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_59/zeros/mulm
gru_59/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru_59/zeros/Less/y
gru_59/zeros/LessLessgru_59/zeros/mul:z:0gru_59/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_59/zeros/Lessp
gru_59/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_59/zeros/packed/1
gru_59/zeros/packedPackgru_59/strided_slice:output:0gru_59/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_59/zeros/packedm
gru_59/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_59/zeros/Const
gru_59/zerosFillgru_59/zeros/packed:output:0gru_59/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/zeros
gru_59/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_59/transpose/perm
gru_59/transpose	Transposeinputsgru_59/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru_59/transposed
gru_59/Shape_1Shapegru_59/transpose:y:0*
T0*
_output_shapes
:2
gru_59/Shape_1
gru_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice_1/stack
gru_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_1/stack_1
gru_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_1/stack_2
gru_59/strided_slice_1StridedSlicegru_59/Shape_1:output:0%gru_59/strided_slice_1/stack:output:0'gru_59/strided_slice_1/stack_1:output:0'gru_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_59/strided_slice_1
"gru_59/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_59/TensorArrayV2/element_shapeЮ
gru_59/TensorArrayV2TensorListReserve+gru_59/TensorArrayV2/element_shape:output:0gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_59/TensorArrayV2Э
<gru_59/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2>
<gru_59/TensorArrayUnstack/TensorListFromTensor/element_shape
.gru_59/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_59/transpose:y:0Egru_59/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_59/TensorArrayUnstack/TensorListFromTensor
gru_59/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice_2/stack
gru_59/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_2/stack_1
gru_59/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_2/stack_2І
gru_59/strided_slice_2StridedSlicegru_59/transpose:y:0%gru_59/strided_slice_2/stack:output:0'gru_59/strided_slice_2/stack_1:output:0'gru_59/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
gru_59/strided_slice_2В
!gru_59/gru_cell_59/ReadVariableOpReadVariableOp*gru_59_gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_59/gru_cell_59/ReadVariableOpЅ
gru_59/gru_cell_59/unstackUnpack)gru_59/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_59/gru_cell_59/unstackЧ
(gru_59/gru_cell_59/MatMul/ReadVariableOpReadVariableOp1gru_59_gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(gru_59/gru_cell_59/MatMul/ReadVariableOpЦ
gru_59/gru_cell_59/MatMulMatMulgru_59/strided_slice_2:output:00gru_59/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/MatMulР
gru_59/gru_cell_59/BiasAddBiasAdd#gru_59/gru_cell_59/MatMul:product:0#gru_59/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/BiasAddv
gru_59/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/gru_cell_59/Const
"gru_59/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_59/gru_cell_59/split/split_dimј
gru_59/gru_cell_59/splitSplit+gru_59/gru_cell_59/split/split_dim:output:0#gru_59/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_59/gru_cell_59/splitЭ
*gru_59/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp3gru_59_gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02,
*gru_59/gru_cell_59/MatMul_1/ReadVariableOpТ
gru_59/gru_cell_59/MatMul_1MatMulgru_59/zeros:output:02gru_59/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/MatMul_1Ц
gru_59/gru_cell_59/BiasAdd_1BiasAdd%gru_59/gru_cell_59/MatMul_1:product:0#gru_59/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/BiasAdd_1
gru_59/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_59/gru_cell_59/Const_1
$gru_59/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$gru_59/gru_cell_59/split_1/split_dimВ
gru_59/gru_cell_59/split_1SplitV%gru_59/gru_cell_59/BiasAdd_1:output:0#gru_59/gru_cell_59/Const_1:output:0-gru_59/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_59/gru_cell_59/split_1Г
gru_59/gru_cell_59/addAddV2!gru_59/gru_cell_59/split:output:0#gru_59/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add
gru_59/gru_cell_59/SigmoidSigmoidgru_59/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/SigmoidЗ
gru_59/gru_cell_59/add_1AddV2!gru_59/gru_cell_59/split:output:1#gru_59/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_1
gru_59/gru_cell_59/Sigmoid_1Sigmoidgru_59/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/Sigmoid_1А
gru_59/gru_cell_59/mulMul gru_59/gru_cell_59/Sigmoid_1:y:0#gru_59/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mulЎ
gru_59/gru_cell_59/add_2AddV2!gru_59/gru_cell_59/split:output:2gru_59/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_2
gru_59/gru_cell_59/ReluRelugru_59/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/ReluЄ
gru_59/gru_cell_59/mul_1Mulgru_59/gru_cell_59/Sigmoid:y:0gru_59/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mul_1y
gru_59/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_59/gru_cell_59/sub/xЌ
gru_59/gru_cell_59/subSub!gru_59/gru_cell_59/sub/x:output:0gru_59/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/subА
gru_59/gru_cell_59/mul_2Mulgru_59/gru_cell_59/sub:z:0%gru_59/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mul_2Ћ
gru_59/gru_cell_59/add_3AddV2gru_59/gru_cell_59/mul_1:z:0gru_59/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_3
$gru_59/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2&
$gru_59/TensorArrayV2_1/element_shapeд
gru_59/TensorArrayV2_1TensorListReserve-gru_59/TensorArrayV2_1/element_shape:output:0gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_59/TensorArrayV2_1\
gru_59/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_59/time
gru_59/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
gru_59/while/maximum_iterationsx
gru_59/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_59/while/loop_counter
gru_59/whileWhile"gru_59/while/loop_counter:output:0(gru_59/while/maximum_iterations:output:0gru_59/time:output:0gru_59/TensorArrayV2_1:handle:0gru_59/zeros:output:0gru_59/strided_slice_1:output:0>gru_59/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_59_gru_cell_59_readvariableop_resource1gru_59_gru_cell_59_matmul_readvariableop_resource3gru_59_gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*&
bodyR
gru_59_while_body_50188224*&
condR
gru_59_while_cond_50188223*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
gru_59/whileУ
7gru_59/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   29
7gru_59/TensorArrayV2Stack/TensorListStack/element_shape
)gru_59/TensorArrayV2Stack/TensorListStackTensorListStackgru_59/while:output:3@gru_59/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02+
)gru_59/TensorArrayV2Stack/TensorListStack
gru_59/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
gru_59/strided_slice_3/stack
gru_59/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_59/strided_slice_3/stack_1
gru_59/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_3/stack_2Ф
gru_59/strided_slice_3StridedSlice2gru_59/TensorArrayV2Stack/TensorListStack:tensor:0%gru_59/strided_slice_3/stack:output:0'gru_59/strided_slice_3/stack_1:output:0'gru_59/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
gru_59/strided_slice_3
gru_59/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_59/transpose_1/permЪ
gru_59/transpose_1	Transpose2gru_59/TensorArrayV2Stack/TensorListStack:tensor:0 gru_59/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
gru_59/transpose_1t
gru_59/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_59/runtime
dropout_68/IdentityIdentitylstm_68/transpose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_68/Identity
dropout_69/IdentityIdentitygru_59/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_69/Identityj
lstm_69/ShapeShapedropout_68/Identity:output:0*
T0*
_output_shapes
:2
lstm_69/Shape
lstm_69/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice/stack
lstm_69/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_69/strided_slice/stack_1
lstm_69/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_69/strided_slice/stack_2
lstm_69/strided_sliceStridedSlicelstm_69/Shape:output:0$lstm_69/strided_slice/stack:output:0&lstm_69/strided_slice/stack_1:output:0&lstm_69/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_69/strided_slicel
lstm_69/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros/mul/y
lstm_69/zeros/mulMullstm_69/strided_slice:output:0lstm_69/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros/mulo
lstm_69/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_69/zeros/Less/y
lstm_69/zeros/LessLesslstm_69/zeros/mul:z:0lstm_69/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros/Lessr
lstm_69/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros/packed/1Ѓ
lstm_69/zeros/packedPacklstm_69/strided_slice:output:0lstm_69/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_69/zeros/packedo
lstm_69/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/zeros/Const
lstm_69/zerosFilllstm_69/zeros/packed:output:0lstm_69/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/zerosp
lstm_69/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros_1/mul/y
lstm_69/zeros_1/mulMullstm_69/strided_slice:output:0lstm_69/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros_1/muls
lstm_69/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_69/zeros_1/Less/y
lstm_69/zeros_1/LessLesslstm_69/zeros_1/mul:z:0lstm_69/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros_1/Lessv
lstm_69/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros_1/packed/1Љ
lstm_69/zeros_1/packedPacklstm_69/strided_slice:output:0!lstm_69/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_69/zeros_1/packeds
lstm_69/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/zeros_1/Const
lstm_69/zeros_1Filllstm_69/zeros_1/packed:output:0lstm_69/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/zeros_1
lstm_69/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_69/transpose/permБ
lstm_69/transpose	Transposedropout_68/Identity:output:0lstm_69/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_69/transposeg
lstm_69/Shape_1Shapelstm_69/transpose:y:0*
T0*
_output_shapes
:2
lstm_69/Shape_1
lstm_69/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice_1/stack
lstm_69/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_1/stack_1
lstm_69/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_1/stack_2
lstm_69/strided_slice_1StridedSlicelstm_69/Shape_1:output:0&lstm_69/strided_slice_1/stack:output:0(lstm_69/strided_slice_1/stack_1:output:0(lstm_69/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_69/strided_slice_1
#lstm_69/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_69/TensorArrayV2/element_shapeв
lstm_69/TensorArrayV2TensorListReserve,lstm_69/TensorArrayV2/element_shape:output:0 lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_69/TensorArrayV2Я
=lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2?
=lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_69/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_69/transpose:y:0Flstm_69/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_69/TensorArrayUnstack/TensorListFromTensor
lstm_69/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice_2/stack
lstm_69/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_2/stack_1
lstm_69/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_2/stack_2Ќ
lstm_69/strided_slice_2StridedSlicelstm_69/transpose:y:0&lstm_69/strided_slice_2/stack:output:0(lstm_69/strided_slice_2/stack_1:output:0(lstm_69/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_69/strided_slice_2Э
*lstm_69/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3lstm_69_lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02,
*lstm_69/lstm_cell_69/MatMul/ReadVariableOpЭ
lstm_69/lstm_cell_69/MatMulMatMul lstm_69/strided_slice_2:output:02lstm_69/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/MatMulг
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02.
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpЩ
lstm_69/lstm_cell_69/MatMul_1MatMullstm_69/zeros:output:04lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/MatMul_1Р
lstm_69/lstm_cell_69/addAddV2%lstm_69/lstm_cell_69/MatMul:product:0'lstm_69/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/addЬ
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpЭ
lstm_69/lstm_cell_69/BiasAddBiasAddlstm_69/lstm_cell_69/add:z:03lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/BiasAddz
lstm_69/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/lstm_cell_69/Const
$lstm_69/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_69/lstm_cell_69/split/split_dim
lstm_69/lstm_cell_69/splitSplit-lstm_69/lstm_cell_69/split/split_dim:output:0%lstm_69/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_69/lstm_cell_69/split
lstm_69/lstm_cell_69/SigmoidSigmoid#lstm_69/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/SigmoidЂ
lstm_69/lstm_cell_69/Sigmoid_1Sigmoid#lstm_69/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/lstm_cell_69/Sigmoid_1Ћ
lstm_69/lstm_cell_69/mulMul"lstm_69/lstm_cell_69/Sigmoid_1:y:0lstm_69/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul
lstm_69/lstm_cell_69/ReluRelu#lstm_69/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/ReluМ
lstm_69/lstm_cell_69/mul_1Mul lstm_69/lstm_cell_69/Sigmoid:y:0'lstm_69/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul_1Б
lstm_69/lstm_cell_69/add_1AddV2lstm_69/lstm_cell_69/mul:z:0lstm_69/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/add_1Ђ
lstm_69/lstm_cell_69/Sigmoid_2Sigmoid#lstm_69/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/lstm_cell_69/Sigmoid_2
lstm_69/lstm_cell_69/Relu_1Relulstm_69/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/Relu_1Р
lstm_69/lstm_cell_69/mul_2Mul"lstm_69/lstm_cell_69/Sigmoid_2:y:0)lstm_69/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul_2
%lstm_69/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2'
%lstm_69/TensorArrayV2_1/element_shapeи
lstm_69/TensorArrayV2_1TensorListReserve.lstm_69/TensorArrayV2_1/element_shape:output:0 lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_69/TensorArrayV2_1^
lstm_69/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_69/time
 lstm_69/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_69/while/maximum_iterationsz
lstm_69/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_69/while/loop_counterъ
lstm_69/whileWhile#lstm_69/while/loop_counter:output:0)lstm_69/while/maximum_iterations:output:0lstm_69/time:output:0 lstm_69/TensorArrayV2_1:handle:0lstm_69/zeros:output:0lstm_69/zeros_1:output:0 lstm_69/strided_slice_1:output:0?lstm_69/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_69_lstm_cell_69_matmul_readvariableop_resource5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource4lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_69_while_body_50188380*'
condR
lstm_69_while_cond_50188379*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
lstm_69/whileХ
8lstm_69/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2:
8lstm_69/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_69/TensorArrayV2Stack/TensorListStackTensorListStacklstm_69/while:output:3Alstm_69/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02,
*lstm_69/TensorArrayV2Stack/TensorListStack
lstm_69/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_69/strided_slice_3/stack
lstm_69/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_69/strided_slice_3/stack_1
lstm_69/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_3/stack_2Ъ
lstm_69/strided_slice_3StridedSlice3lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_69/strided_slice_3/stack:output:0(lstm_69/strided_slice_3/stack_1:output:0(lstm_69/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
lstm_69/strided_slice_3
lstm_69/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_69/transpose_1/permЮ
lstm_69/transpose_1	Transpose3lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_69/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
lstm_69/transpose_1v
lstm_69/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/runtimeЋ
dense_327/MatMul/ReadVariableOpReadVariableOp(dense_327_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02!
dense_327/MatMul/ReadVariableOpЋ
dense_327/MatMulMatMul lstm_69/strided_slice_3:output:0'dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/MatMulЊ
 dense_327/BiasAdd/ReadVariableOpReadVariableOp)dense_327_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_327/BiasAdd/ReadVariableOpЉ
dense_327/BiasAddBiasAdddense_327/MatMul:product:0(dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/BiasAddv
dense_327/ReluReludense_327/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/ReluЋ
dense_328/MatMul/ReadVariableOpReadVariableOp(dense_328_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02!
dense_328/MatMul/ReadVariableOpЇ
dense_328/MatMulMatMuldropout_69/Identity:output:0'dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/MatMulЊ
 dense_328/BiasAdd/ReadVariableOpReadVariableOp)dense_328_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_328/BiasAdd/ReadVariableOpЉ
dense_328/BiasAddBiasAdddense_328/MatMul:product:0(dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/BiasAddv
dense_328/ReluReludense_328/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/Relux
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axisг
concatenate_9/concatConcatV2dense_327/Relu:activations:0dense_328/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatenate_9/concatЋ
dense_329/MatMul/ReadVariableOpReadVariableOp(dense_329_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02!
dense_329/MatMul/ReadVariableOpЈ
dense_329/MatMulMatMulconcatenate_9/concat:output:0'dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_329/MatMulЊ
 dense_329/BiasAdd/ReadVariableOpReadVariableOp)dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_329/BiasAdd/ReadVariableOpЉ
dense_329/BiasAddBiasAdddense_329/MatMul:product:0(dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_329/BiasAddќ
IdentityIdentitydense_329/BiasAdd:output:0!^dense_327/BiasAdd/ReadVariableOp ^dense_327/MatMul/ReadVariableOp!^dense_328/BiasAdd/ReadVariableOp ^dense_328/MatMul/ReadVariableOp!^dense_329/BiasAdd/ReadVariableOp ^dense_329/MatMul/ReadVariableOp)^gru_59/gru_cell_59/MatMul/ReadVariableOp+^gru_59/gru_cell_59/MatMul_1/ReadVariableOp"^gru_59/gru_cell_59/ReadVariableOp^gru_59/while,^lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp+^lstm_68/lstm_cell_68/MatMul/ReadVariableOp-^lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp^lstm_68/while,^lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp+^lstm_69/lstm_cell_69/MatMul/ReadVariableOp-^lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp^lstm_69/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_327/BiasAdd/ReadVariableOp dense_327/BiasAdd/ReadVariableOp2B
dense_327/MatMul/ReadVariableOpdense_327/MatMul/ReadVariableOp2D
 dense_328/BiasAdd/ReadVariableOp dense_328/BiasAdd/ReadVariableOp2B
dense_328/MatMul/ReadVariableOpdense_328/MatMul/ReadVariableOp2D
 dense_329/BiasAdd/ReadVariableOp dense_329/BiasAdd/ReadVariableOp2B
dense_329/MatMul/ReadVariableOpdense_329/MatMul/ReadVariableOp2T
(gru_59/gru_cell_59/MatMul/ReadVariableOp(gru_59/gru_cell_59/MatMul/ReadVariableOp2X
*gru_59/gru_cell_59/MatMul_1/ReadVariableOp*gru_59/gru_cell_59/MatMul_1/ReadVariableOp2F
!gru_59/gru_cell_59/ReadVariableOp!gru_59/gru_cell_59/ReadVariableOp2
gru_59/whilegru_59/while2Z
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp2X
*lstm_68/lstm_cell_68/MatMul/ReadVariableOp*lstm_68/lstm_cell_68/MatMul/ReadVariableOp2\
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp2
lstm_68/whilelstm_68/while2Z
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp2X
*lstm_69/lstm_cell_69/MatMul/ReadVariableOp*lstm_69/lstm_cell_69/MatMul/ReadVariableOp2\
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp2
lstm_69/whilelstm_69/while:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
Д
while_cond_50186494
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50186494___redundant_placeholder06
2while_while_cond_50186494___redundant_placeholder16
2while_while_cond_50186494___redundant_placeholder26
2while_while_cond_50186494___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ы
f
H__inference_dropout_69_layer_call_and_return_conditional_losses_50186821

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Х[
є
E__inference_lstm_69_layer_call_and_return_conditional_losses_50187139

inputs/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50187054*
condR
while_cond_50187053*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Е
Э
while_cond_50188777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50188777___redundant_placeholder06
2while_while_cond_50188777___redundant_placeholder16
2while_while_cond_50188777___redundant_placeholder26
2while_while_cond_50188777___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
C

while_body_50190469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Є

Ц
*__inference_model_9_layer_call_fn_50188557

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_501874332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б.
д
E__inference_model_9_layer_call_and_return_conditional_losses_50187266
input_10
lstm_68_50186420
lstm_68_50186422
lstm_68_50186424
gru_59_50186767
gru_59_50186769
gru_59_50186771
lstm_69_50187162
lstm_69_50187164
lstm_69_50187166
dense_327_50187191
dense_327_50187193
dense_328_50187218
dense_328_50187220
dense_329_50187260
dense_329_50187262
identityЂ!dense_327/StatefulPartitionedCallЂ!dense_328/StatefulPartitionedCallЂ!dense_329/StatefulPartitionedCallЂ"dropout_68/StatefulPartitionedCallЂ"dropout_69/StatefulPartitionedCallЂgru_59/StatefulPartitionedCallЂlstm_68/StatefulPartitionedCallЂlstm_69/StatefulPartitionedCallЛ
lstm_68/StatefulPartitionedCallStatefulPartitionedCallinput_10lstm_68_50186420lstm_68_50186422lstm_68_50186424*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501862442!
lstm_68/StatefulPartitionedCallЈ
gru_59/StatefulPartitionedCallStatefulPartitionedCallinput_10gru_59_50186767gru_59_50186769gru_59_50186771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501865852 
gru_59/StatefulPartitionedCallІ
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall(lstm_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867862$
"dropout_68/StatefulPartitionedCallН
"dropout_69/StatefulPartitionedCallStatefulPartitionedCall'gru_59/StatefulPartitionedCall:output:0#^dropout_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868162$
"dropout_69/StatefulPartitionedCallб
lstm_69/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0lstm_69_50187162lstm_69_50187164lstm_69_50187166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501869862!
lstm_69/StatefulPartitionedCallФ
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(lstm_69/StatefulPartitionedCall:output:0dense_327_50187191dense_327_50187193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_327_layer_call_and_return_conditional_losses_501871802#
!dense_327/StatefulPartitionedCallЧ
!dense_328/StatefulPartitionedCallStatefulPartitionedCall+dropout_69/StatefulPartitionedCall:output:0dense_328_50187218dense_328_50187220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_328_layer_call_and_return_conditional_losses_501872072#
!dense_328/StatefulPartitionedCallЙ
concatenate_9/PartitionedCallPartitionedCall*dense_327/StatefulPartitionedCall:output:0*dense_328/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_501872302
concatenate_9/PartitionedCallТ
!dense_329/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_329_50187260dense_329_50187262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_329_layer_call_and_return_conditional_losses_501872492#
!dense_329/StatefulPartitionedCall
IdentityIdentity*dense_329/StatefulPartitionedCall:output:0"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall#^dropout_69/StatefulPartitionedCall^gru_59/StatefulPartitionedCall ^lstm_68/StatefulPartitionedCall ^lstm_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2H
"dropout_69/StatefulPartitionedCall"dropout_69/StatefulPartitionedCall2@
gru_59/StatefulPartitionedCallgru_59/StatefulPartitionedCall2B
lstm_68/StatefulPartitionedCalllstm_68/StatefulPartitionedCall2B
lstm_69/StatefulPartitionedCalllstm_69/StatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
њG
А
while_body_50186495
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_50186653
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50186653___redundant_placeholder06
2while_while_cond_50186653___redundant_placeholder16
2while_while_cond_50186653___redundant_placeholder26
2while_while_cond_50186653___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
O


lstm_68_while_body_50188074,
(lstm_68_while_lstm_68_while_loop_counter2
.lstm_68_while_lstm_68_while_maximum_iterations
lstm_68_while_placeholder
lstm_68_while_placeholder_1
lstm_68_while_placeholder_2
lstm_68_while_placeholder_3+
'lstm_68_while_lstm_68_strided_slice_1_0g
clstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0A
=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0@
<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0
lstm_68_while_identity
lstm_68_while_identity_1
lstm_68_while_identity_2
lstm_68_while_identity_3
lstm_68_while_identity_4
lstm_68_while_identity_5)
%lstm_68_while_lstm_68_strided_slice_1e
alstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor=
9lstm_68_while_lstm_cell_68_matmul_readvariableop_resource?
;lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource>
:lstm_68_while_lstm_cell_68_biasadd_readvariableop_resourceЂ1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpЂ0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpЂ2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpг
?lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_68/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0lstm_68_while_placeholderHlstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_68/while/TensorArrayV2Read/TensorListGetItemс
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype022
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpї
!lstm_68/while/lstm_cell_68/MatMulMatMul8lstm_68/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!lstm_68/while/lstm_cell_68/MatMulч
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype024
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpр
#lstm_68/while/lstm_cell_68/MatMul_1MatMullstm_68_while_placeholder_2:lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#lstm_68/while/lstm_cell_68/MatMul_1и
lstm_68/while/lstm_cell_68/addAddV2+lstm_68/while/lstm_cell_68/MatMul:product:0-lstm_68/while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm_68/while/lstm_cell_68/addр
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype023
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpх
"lstm_68/while/lstm_cell_68/BiasAddBiasAdd"lstm_68/while/lstm_cell_68/add:z:09lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_68/while/lstm_cell_68/BiasAdd
 lstm_68/while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_68/while/lstm_cell_68/Const
*lstm_68/while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_68/while/lstm_cell_68/split/split_dimЋ
 lstm_68/while/lstm_cell_68/splitSplit3lstm_68/while/lstm_cell_68/split/split_dim:output:0+lstm_68/while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2"
 lstm_68/while/lstm_cell_68/splitА
"lstm_68/while/lstm_cell_68/SigmoidSigmoid)lstm_68/while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"lstm_68/while/lstm_cell_68/SigmoidД
$lstm_68/while/lstm_cell_68/Sigmoid_1Sigmoid)lstm_68/while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_68/while/lstm_cell_68/Sigmoid_1Р
lstm_68/while/lstm_cell_68/mulMul(lstm_68/while/lstm_cell_68/Sigmoid_1:y:0lstm_68_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/while/lstm_cell_68/mulЇ
lstm_68/while/lstm_cell_68/ReluRelu)lstm_68/while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2!
lstm_68/while/lstm_cell_68/Reluд
 lstm_68/while/lstm_cell_68/mul_1Mul&lstm_68/while/lstm_cell_68/Sigmoid:y:0-lstm_68/while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/mul_1Щ
 lstm_68/while/lstm_cell_68/add_1AddV2"lstm_68/while/lstm_cell_68/mul:z:0$lstm_68/while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/add_1Д
$lstm_68/while/lstm_cell_68/Sigmoid_2Sigmoid)lstm_68/while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_68/while/lstm_cell_68/Sigmoid_2І
!lstm_68/while/lstm_cell_68/Relu_1Relu$lstm_68/while/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2#
!lstm_68/while/lstm_cell_68/Relu_1и
 lstm_68/while/lstm_cell_68/mul_2Mul(lstm_68/while/lstm_cell_68/Sigmoid_2:y:0/lstm_68/while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_68/while/lstm_cell_68/mul_2
2lstm_68/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_68_while_placeholder_1lstm_68_while_placeholder$lstm_68/while/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_68/while/TensorArrayV2Write/TensorListSetIteml
lstm_68/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/while/add/y
lstm_68/while/addAddV2lstm_68_while_placeholderlstm_68/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_68/while/addp
lstm_68/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/while/add_1/y
lstm_68/while/add_1AddV2(lstm_68_while_lstm_68_while_loop_counterlstm_68/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_68/while/add_1
lstm_68/while/IdentityIdentitylstm_68/while/add_1:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity­
lstm_68/while/Identity_1Identity.lstm_68_while_lstm_68_while_maximum_iterations2^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_1
lstm_68/while/Identity_2Identitylstm_68/while/add:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_2С
lstm_68/while/Identity_3IdentityBlstm_68/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_68/while/Identity_3Д
lstm_68/while/Identity_4Identity$lstm_68/while/lstm_cell_68/mul_2:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/while/Identity_4Д
lstm_68/while/Identity_5Identity$lstm_68/while/lstm_cell_68/add_1:z:02^lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1^lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp3^lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/while/Identity_5"9
lstm_68_while_identitylstm_68/while/Identity:output:0"=
lstm_68_while_identity_1!lstm_68/while/Identity_1:output:0"=
lstm_68_while_identity_2!lstm_68/while/Identity_2:output:0"=
lstm_68_while_identity_3!lstm_68/while/Identity_3:output:0"=
lstm_68_while_identity_4!lstm_68/while/Identity_4:output:0"=
lstm_68_while_identity_5!lstm_68/while/Identity_5:output:0"P
%lstm_68_while_lstm_68_strided_slice_1'lstm_68_while_lstm_68_strided_slice_1_0"z
:lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource<lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0"|
;lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource=lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0"x
9lstm_68_while_lstm_cell_68_matmul_readvariableop_resource;lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0"Ш
alstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensorclstm_68_while_tensorarrayv2read_tensorlistgetitem_lstm_68_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2f
1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp1lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp2d
0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp0lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp2h
2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp2lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 


*__inference_lstm_69_layer_call_fn_50190576

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501871392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs

I
-__inference_dropout_69_layer_call_fn_50190603

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868212
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Р%

while_body_50184706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_68_50184730_0!
while_lstm_cell_68_50184732_0!
while_lstm_cell_68_50184734_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_68_50184730
while_lstm_cell_68_50184732
while_lstm_cell_68_50184734Ђ*while/lstm_cell_68/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_68/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_68_50184730_0while_lstm_cell_68_50184732_0while_lstm_cell_68_50184734_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501843792,
*while/lstm_cell_68/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_68/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_68/StatefulPartitionedCall:output:1+^while/lstm_cell_68/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_68/StatefulPartitionedCall:output:2+^while/lstm_cell_68/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_68_50184730while_lstm_cell_68_50184730_0"<
while_lstm_cell_68_50184732while_lstm_cell_68_50184732_0"<
while_lstm_cell_68_50184734while_lstm_cell_68_50184734_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2X
*while/lstm_cell_68/StatefulPartitionedCall*while/lstm_cell_68/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 

g
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190588

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ц

,__inference_dense_327_layer_call_fn_50190623

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_327_layer_call_and_return_conditional_losses_501871802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Р%

while_body_50184838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_68_50184862_0!
while_lstm_cell_68_50184864_0!
while_lstm_cell_68_50184866_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_68_50184862
while_lstm_cell_68_50184864
while_lstm_cell_68_50184866Ђ*while/lstm_cell_68/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_68/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_68_50184862_0while_lstm_cell_68_50184864_0while_lstm_cell_68_50184866_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501844122,
*while/lstm_cell_68/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_68/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_68/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_68/StatefulPartitionedCall:output:1+^while/lstm_cell_68/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_68/StatefulPartitionedCall:output:2+^while/lstm_cell_68/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_68_50184862while_lstm_cell_68_50184862_0"<
while_lstm_cell_68_50184864while_lstm_cell_68_50184864_0"<
while_lstm_cell_68_50184866while_lstm_cell_68_50184866_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2X
*while/lstm_cell_68/StatefulPartitionedCall*while/lstm_cell_68/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_50189308
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50189308___redundant_placeholder06
2while_while_cond_50189308___redundant_placeholder16
2while_while_cond_50189308___redundant_placeholder26
2while_while_cond_50189308___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Х[
є
E__inference_lstm_69_layer_call_and_return_conditional_losses_50186986

inputs/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50186901*
condR
while_cond_50186900*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Е
Э
while_cond_50190468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50190468___redundant_placeholder06
2while_while_cond_50190468___redundant_placeholder16
2while_while_cond_50190468___redundant_placeholder26
2while_while_cond_50190468___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:


*__inference_lstm_69_layer_call_fn_50190565

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501869862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
­
н
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50184379

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates
Е
п
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190916

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
џZ

#model_9_lstm_68_while_body_50183893<
8model_9_lstm_68_while_model_9_lstm_68_while_loop_counterB
>model_9_lstm_68_while_model_9_lstm_68_while_maximum_iterations%
!model_9_lstm_68_while_placeholder'
#model_9_lstm_68_while_placeholder_1'
#model_9_lstm_68_while_placeholder_2'
#model_9_lstm_68_while_placeholder_3;
7model_9_lstm_68_while_model_9_lstm_68_strided_slice_1_0w
smodel_9_lstm_68_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_68_tensorarrayunstack_tensorlistfromtensor_0G
Cmodel_9_lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0I
Emodel_9_lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0H
Dmodel_9_lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0"
model_9_lstm_68_while_identity$
 model_9_lstm_68_while_identity_1$
 model_9_lstm_68_while_identity_2$
 model_9_lstm_68_while_identity_3$
 model_9_lstm_68_while_identity_4$
 model_9_lstm_68_while_identity_59
5model_9_lstm_68_while_model_9_lstm_68_strided_slice_1u
qmodel_9_lstm_68_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_68_tensorarrayunstack_tensorlistfromtensorE
Amodel_9_lstm_68_while_lstm_cell_68_matmul_readvariableop_resourceG
Cmodel_9_lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resourceF
Bmodel_9_lstm_68_while_lstm_cell_68_biasadd_readvariableop_resourceЂ9model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpЂ8model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpЂ:model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpу
Gmodel_9/lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gmodel_9/lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9model_9/lstm_68/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_9_lstm_68_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_68_tensorarrayunstack_tensorlistfromtensor_0!model_9_lstm_68_while_placeholderPmodel_9/lstm_68/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9model_9/lstm_68/while/TensorArrayV2Read/TensorListGetItemљ
8model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOpCmodel_9_lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02:
8model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp
)model_9/lstm_68/while/lstm_cell_68/MatMulMatMul@model_9/lstm_68/while/TensorArrayV2Read/TensorListGetItem:item:0@model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2+
)model_9/lstm_68/while/lstm_cell_68/MatMulџ
:model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOpEmodel_9_lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02<
:model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp
+model_9/lstm_68/while/lstm_cell_68/MatMul_1MatMul#model_9_lstm_68_while_placeholder_2Bmodel_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2-
+model_9/lstm_68/while/lstm_cell_68/MatMul_1ј
&model_9/lstm_68/while/lstm_cell_68/addAddV23model_9/lstm_68/while/lstm_cell_68/MatMul:product:05model_9/lstm_68/while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&model_9/lstm_68/while/lstm_cell_68/addј
9model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02;
9model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp
*model_9/lstm_68/while/lstm_cell_68/BiasAddBiasAdd*model_9/lstm_68/while/lstm_cell_68/add:z:0Amodel_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2,
*model_9/lstm_68/while/lstm_cell_68/BiasAdd
(model_9/lstm_68/while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_9/lstm_68/while/lstm_cell_68/ConstЊ
2model_9/lstm_68/while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/lstm_68/while/lstm_cell_68/split/split_dimЫ
(model_9/lstm_68/while/lstm_cell_68/splitSplit;model_9/lstm_68/while/lstm_cell_68/split/split_dim:output:03model_9/lstm_68/while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2*
(model_9/lstm_68/while/lstm_cell_68/splitШ
*model_9/lstm_68/while/lstm_cell_68/SigmoidSigmoid1model_9/lstm_68/while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2,
*model_9/lstm_68/while/lstm_cell_68/SigmoidЬ
,model_9/lstm_68/while/lstm_cell_68/Sigmoid_1Sigmoid1model_9/lstm_68/while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2.
,model_9/lstm_68/while/lstm_cell_68/Sigmoid_1р
&model_9/lstm_68/while/lstm_cell_68/mulMul0model_9/lstm_68/while/lstm_cell_68/Sigmoid_1:y:0#model_9_lstm_68_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_68/while/lstm_cell_68/mulП
'model_9/lstm_68/while/lstm_cell_68/ReluRelu1model_9/lstm_68/while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2)
'model_9/lstm_68/while/lstm_cell_68/Reluє
(model_9/lstm_68/while/lstm_cell_68/mul_1Mul.model_9/lstm_68/while/lstm_cell_68/Sigmoid:y:05model_9/lstm_68/while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_68/while/lstm_cell_68/mul_1щ
(model_9/lstm_68/while/lstm_cell_68/add_1AddV2*model_9/lstm_68/while/lstm_cell_68/mul:z:0,model_9/lstm_68/while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_68/while/lstm_cell_68/add_1Ь
,model_9/lstm_68/while/lstm_cell_68/Sigmoid_2Sigmoid1model_9/lstm_68/while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2.
,model_9/lstm_68/while/lstm_cell_68/Sigmoid_2О
)model_9/lstm_68/while/lstm_cell_68/Relu_1Relu,model_9/lstm_68/while/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2+
)model_9/lstm_68/while/lstm_cell_68/Relu_1ј
(model_9/lstm_68/while/lstm_cell_68/mul_2Mul0model_9/lstm_68/while/lstm_cell_68/Sigmoid_2:y:07model_9/lstm_68/while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_68/while/lstm_cell_68/mul_2А
:model_9/lstm_68/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_9_lstm_68_while_placeholder_1!model_9_lstm_68_while_placeholder,model_9/lstm_68/while/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_9/lstm_68/while/TensorArrayV2Write/TensorListSetItem|
model_9/lstm_68/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_68/while/add/yЉ
model_9/lstm_68/while/addAddV2!model_9_lstm_68_while_placeholder$model_9/lstm_68/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/while/add
model_9/lstm_68/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_68/while/add_1/yЦ
model_9/lstm_68/while/add_1AddV28model_9_lstm_68_while_model_9_lstm_68_while_loop_counter&model_9/lstm_68/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_68/while/add_1Т
model_9/lstm_68/while/IdentityIdentitymodel_9/lstm_68/while/add_1:z:0:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/lstm_68/while/Identityх
 model_9/lstm_68/while/Identity_1Identity>model_9_lstm_68_while_model_9_lstm_68_while_maximum_iterations:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_68/while/Identity_1Ф
 model_9/lstm_68/while/Identity_2Identitymodel_9/lstm_68/while/add:z:0:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_68/while/Identity_2ё
 model_9/lstm_68/while/Identity_3IdentityJmodel_9/lstm_68/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_68/while/Identity_3ф
 model_9/lstm_68/while/Identity_4Identity,model_9/lstm_68/while/lstm_cell_68/mul_2:z:0:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_68/while/Identity_4ф
 model_9/lstm_68/while/Identity_5Identity,model_9/lstm_68/while/lstm_cell_68/add_1:z:0:^model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9^model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp;^model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_68/while/Identity_5"I
model_9_lstm_68_while_identity'model_9/lstm_68/while/Identity:output:0"M
 model_9_lstm_68_while_identity_1)model_9/lstm_68/while/Identity_1:output:0"M
 model_9_lstm_68_while_identity_2)model_9/lstm_68/while/Identity_2:output:0"M
 model_9_lstm_68_while_identity_3)model_9/lstm_68/while/Identity_3:output:0"M
 model_9_lstm_68_while_identity_4)model_9/lstm_68/while/Identity_4:output:0"M
 model_9_lstm_68_while_identity_5)model_9/lstm_68/while/Identity_5:output:0"
Bmodel_9_lstm_68_while_lstm_cell_68_biasadd_readvariableop_resourceDmodel_9_lstm_68_while_lstm_cell_68_biasadd_readvariableop_resource_0"
Cmodel_9_lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resourceEmodel_9_lstm_68_while_lstm_cell_68_matmul_1_readvariableop_resource_0"
Amodel_9_lstm_68_while_lstm_cell_68_matmul_readvariableop_resourceCmodel_9_lstm_68_while_lstm_cell_68_matmul_readvariableop_resource_0"p
5model_9_lstm_68_while_model_9_lstm_68_strided_slice_17model_9_lstm_68_while_model_9_lstm_68_strided_slice_1_0"ш
qmodel_9_lstm_68_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_68_tensorarrayunstack_tensorlistfromtensorsmodel_9_lstm_68_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_68_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2v
9model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp9model_9/lstm_68/while/lstm_cell_68/BiasAdd/ReadVariableOp2t
8model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp8model_9/lstm_68/while/lstm_cell_68/MatMul/ReadVariableOp2x
:model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp:model_9/lstm_68/while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
џZ

#model_9_lstm_69_while_body_50184199<
8model_9_lstm_69_while_model_9_lstm_69_while_loop_counterB
>model_9_lstm_69_while_model_9_lstm_69_while_maximum_iterations%
!model_9_lstm_69_while_placeholder'
#model_9_lstm_69_while_placeholder_1'
#model_9_lstm_69_while_placeholder_2'
#model_9_lstm_69_while_placeholder_3;
7model_9_lstm_69_while_model_9_lstm_69_strided_slice_1_0w
smodel_9_lstm_69_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_69_tensorarrayunstack_tensorlistfromtensor_0G
Cmodel_9_lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0I
Emodel_9_lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0H
Dmodel_9_lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0"
model_9_lstm_69_while_identity$
 model_9_lstm_69_while_identity_1$
 model_9_lstm_69_while_identity_2$
 model_9_lstm_69_while_identity_3$
 model_9_lstm_69_while_identity_4$
 model_9_lstm_69_while_identity_59
5model_9_lstm_69_while_model_9_lstm_69_strided_slice_1u
qmodel_9_lstm_69_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_69_tensorarrayunstack_tensorlistfromtensorE
Amodel_9_lstm_69_while_lstm_cell_69_matmul_readvariableop_resourceG
Cmodel_9_lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resourceF
Bmodel_9_lstm_69_while_lstm_cell_69_biasadd_readvariableop_resourceЂ9model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpЂ8model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpЂ:model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpу
Gmodel_9/lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2I
Gmodel_9/lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9model_9/lstm_69/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_9_lstm_69_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_69_tensorarrayunstack_tensorlistfromtensor_0!model_9_lstm_69_while_placeholderPmodel_9/lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02;
9model_9/lstm_69/while/TensorArrayV2Read/TensorListGetItemљ
8model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOpCmodel_9_lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02:
8model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp
)model_9/lstm_69/while/lstm_cell_69/MatMulMatMul@model_9/lstm_69/while/TensorArrayV2Read/TensorListGetItem:item:0@model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)model_9/lstm_69/while/lstm_cell_69/MatMulџ
:model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOpEmodel_9_lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02<
:model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp
+model_9/lstm_69/while/lstm_cell_69/MatMul_1MatMul#model_9_lstm_69_while_placeholder_2Bmodel_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+model_9/lstm_69/while/lstm_cell_69/MatMul_1ј
&model_9/lstm_69/while/lstm_cell_69/addAddV23model_9/lstm_69/while/lstm_cell_69/MatMul:product:05model_9/lstm_69/while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&model_9/lstm_69/while/lstm_cell_69/addј
9model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02;
9model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp
*model_9/lstm_69/while/lstm_cell_69/BiasAddBiasAdd*model_9/lstm_69/while/lstm_cell_69/add:z:0Amodel_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*model_9/lstm_69/while/lstm_cell_69/BiasAdd
(model_9/lstm_69/while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_9/lstm_69/while/lstm_cell_69/ConstЊ
2model_9/lstm_69/while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/lstm_69/while/lstm_cell_69/split/split_dimЫ
(model_9/lstm_69/while/lstm_cell_69/splitSplit;model_9/lstm_69/while/lstm_cell_69/split/split_dim:output:03model_9/lstm_69/while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2*
(model_9/lstm_69/while/lstm_cell_69/splitШ
*model_9/lstm_69/while/lstm_cell_69/SigmoidSigmoid1model_9/lstm_69/while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*model_9/lstm_69/while/lstm_cell_69/SigmoidЬ
,model_9/lstm_69/while/lstm_cell_69/Sigmoid_1Sigmoid1model_9/lstm_69/while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22.
,model_9/lstm_69/while/lstm_cell_69/Sigmoid_1р
&model_9/lstm_69/while/lstm_cell_69/mulMul0model_9/lstm_69/while/lstm_cell_69/Sigmoid_1:y:0#model_9_lstm_69_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_69/while/lstm_cell_69/mulП
'model_9/lstm_69/while/lstm_cell_69/ReluRelu1model_9/lstm_69/while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22)
'model_9/lstm_69/while/lstm_cell_69/Reluє
(model_9/lstm_69/while/lstm_cell_69/mul_1Mul.model_9/lstm_69/while/lstm_cell_69/Sigmoid:y:05model_9/lstm_69/while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_69/while/lstm_cell_69/mul_1щ
(model_9/lstm_69/while/lstm_cell_69/add_1AddV2*model_9/lstm_69/while/lstm_cell_69/mul:z:0,model_9/lstm_69/while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_69/while/lstm_cell_69/add_1Ь
,model_9/lstm_69/while/lstm_cell_69/Sigmoid_2Sigmoid1model_9/lstm_69/while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22.
,model_9/lstm_69/while/lstm_cell_69/Sigmoid_2О
)model_9/lstm_69/while/lstm_cell_69/Relu_1Relu,model_9/lstm_69/while/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)model_9/lstm_69/while/lstm_cell_69/Relu_1ј
(model_9/lstm_69/while/lstm_cell_69/mul_2Mul0model_9/lstm_69/while/lstm_cell_69/Sigmoid_2:y:07model_9/lstm_69/while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_69/while/lstm_cell_69/mul_2А
:model_9/lstm_69/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_9_lstm_69_while_placeholder_1!model_9_lstm_69_while_placeholder,model_9/lstm_69/while/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_9/lstm_69/while/TensorArrayV2Write/TensorListSetItem|
model_9/lstm_69/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_69/while/add/yЉ
model_9/lstm_69/while/addAddV2!model_9_lstm_69_while_placeholder$model_9/lstm_69/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/while/add
model_9/lstm_69/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_69/while/add_1/yЦ
model_9/lstm_69/while/add_1AddV28model_9_lstm_69_while_model_9_lstm_69_while_loop_counter&model_9/lstm_69/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_69/while/add_1Т
model_9/lstm_69/while/IdentityIdentitymodel_9/lstm_69/while/add_1:z:0:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/lstm_69/while/Identityх
 model_9/lstm_69/while/Identity_1Identity>model_9_lstm_69_while_model_9_lstm_69_while_maximum_iterations:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_69/while/Identity_1Ф
 model_9/lstm_69/while/Identity_2Identitymodel_9/lstm_69/while/add:z:0:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_69/while/Identity_2ё
 model_9/lstm_69/while/Identity_3IdentityJmodel_9/lstm_69/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_69/while/Identity_3ф
 model_9/lstm_69/while/Identity_4Identity,model_9/lstm_69/while/lstm_cell_69/mul_2:z:0:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_69/while/Identity_4ф
 model_9/lstm_69/while/Identity_5Identity,model_9/lstm_69/while/lstm_cell_69/add_1:z:0:^model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9^model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp;^model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_69/while/Identity_5"I
model_9_lstm_69_while_identity'model_9/lstm_69/while/Identity:output:0"M
 model_9_lstm_69_while_identity_1)model_9/lstm_69/while/Identity_1:output:0"M
 model_9_lstm_69_while_identity_2)model_9/lstm_69/while/Identity_2:output:0"M
 model_9_lstm_69_while_identity_3)model_9/lstm_69/while/Identity_3:output:0"M
 model_9_lstm_69_while_identity_4)model_9/lstm_69/while/Identity_4:output:0"M
 model_9_lstm_69_while_identity_5)model_9/lstm_69/while/Identity_5:output:0"
Bmodel_9_lstm_69_while_lstm_cell_69_biasadd_readvariableop_resourceDmodel_9_lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0"
Cmodel_9_lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resourceEmodel_9_lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0"
Amodel_9_lstm_69_while_lstm_cell_69_matmul_readvariableop_resourceCmodel_9_lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0"p
5model_9_lstm_69_while_model_9_lstm_69_strided_slice_17model_9_lstm_69_while_model_9_lstm_69_strided_slice_1_0"ш
qmodel_9_lstm_69_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_69_tensorarrayunstack_tensorlistfromtensorsmodel_9_lstm_69_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_69_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2v
9model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp9model_9/lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp2t
8model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp8model_9/lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp2x
:model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp:model_9/lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_50185877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50185877___redundant_placeholder06
2while_while_cond_50185877___redundant_placeholder16
2while_while_cond_50185877___redundant_placeholder26
2while_while_cond_50185877___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Т
Я
/__inference_lstm_cell_68_layer_call_fn_50190775

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501844122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
х	
А
.__inference_gru_cell_59_layer_call_fn_50190883

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501850282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
	
р
G__inference_dense_329_layer_call_and_return_conditional_losses_50190666

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Е
п
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190708

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
к
Д
while_cond_50185404
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50185404___redundant_placeholder06
2while_while_cond_50185404___redundant_placeholder16
2while_while_cond_50185404___redundant_placeholder26
2while_while_cond_50185404___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Т
Я
/__inference_lstm_cell_69_layer_call_fn_50190983

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1

б
"model_9_gru_59_while_cond_50184042:
6model_9_gru_59_while_model_9_gru_59_while_loop_counter@
<model_9_gru_59_while_model_9_gru_59_while_maximum_iterations$
 model_9_gru_59_while_placeholder&
"model_9_gru_59_while_placeholder_1&
"model_9_gru_59_while_placeholder_2<
8model_9_gru_59_while_less_model_9_gru_59_strided_slice_1T
Pmodel_9_gru_59_while_model_9_gru_59_while_cond_50184042___redundant_placeholder0T
Pmodel_9_gru_59_while_model_9_gru_59_while_cond_50184042___redundant_placeholder1T
Pmodel_9_gru_59_while_model_9_gru_59_while_cond_50184042___redundant_placeholder2T
Pmodel_9_gru_59_while_model_9_gru_59_while_cond_50184042___redundant_placeholder3!
model_9_gru_59_while_identity
Л
model_9/gru_59/while/LessLess model_9_gru_59_while_placeholder8model_9_gru_59_while_less_model_9_gru_59_strided_slice_1*
T0*
_output_shapes
: 2
model_9/gru_59/while/Less
model_9/gru_59/while/IdentityIdentitymodel_9/gru_59/while/Less:z:0*
T0
*
_output_shapes
: 2
model_9/gru_59/while/Identity"G
model_9_gru_59_while_identity&model_9/gru_59/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:


#model_9_lstm_69_while_cond_50184198<
8model_9_lstm_69_while_model_9_lstm_69_while_loop_counterB
>model_9_lstm_69_while_model_9_lstm_69_while_maximum_iterations%
!model_9_lstm_69_while_placeholder'
#model_9_lstm_69_while_placeholder_1'
#model_9_lstm_69_while_placeholder_2'
#model_9_lstm_69_while_placeholder_3>
:model_9_lstm_69_while_less_model_9_lstm_69_strided_slice_1V
Rmodel_9_lstm_69_while_model_9_lstm_69_while_cond_50184198___redundant_placeholder0V
Rmodel_9_lstm_69_while_model_9_lstm_69_while_cond_50184198___redundant_placeholder1V
Rmodel_9_lstm_69_while_model_9_lstm_69_while_cond_50184198___redundant_placeholder2V
Rmodel_9_lstm_69_while_model_9_lstm_69_while_cond_50184198___redundant_placeholder3"
model_9_lstm_69_while_identity
Р
model_9/lstm_69/while/LessLess!model_9_lstm_69_while_placeholder:model_9_lstm_69_while_less_model_9_lstm_69_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_69/while/Less
model_9/lstm_69/while/IdentityIdentitymodel_9/lstm_69/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_9/lstm_69/while/Identity"I
model_9_lstm_69_while_identity'model_9/lstm_69/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к[
п
D__inference_gru_59_layer_call_and_return_conditional_losses_50189739
inputs_0'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50189649*
condR
while_cond_50189648*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ь
Ў
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50185028

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
Ж

*__inference_lstm_68_layer_call_fn_50189213

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501863972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


)__inference_gru_59_layer_call_fn_50189580

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501867442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х[
є
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190401

inputs/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50190316*
condR
while_cond_50190315*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
C

while_body_50186901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
­
н
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50185551

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
М

*__inference_lstm_68_layer_call_fn_50188874
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501847752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

g
H__inference_dropout_69_layer_call_and_return_conditional_losses_50186816

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
М+

E__inference_model_9_layer_call_and_return_conditional_losses_50187433

inputs
lstm_68_50187393
lstm_68_50187395
lstm_68_50187397
gru_59_50187400
gru_59_50187402
gru_59_50187404
lstm_69_50187409
lstm_69_50187411
lstm_69_50187413
dense_327_50187416
dense_327_50187418
dense_328_50187421
dense_328_50187423
dense_329_50187427
dense_329_50187429
identityЂ!dense_327/StatefulPartitionedCallЂ!dense_328/StatefulPartitionedCallЂ!dense_329/StatefulPartitionedCallЂgru_59/StatefulPartitionedCallЂlstm_68/StatefulPartitionedCallЂlstm_69/StatefulPartitionedCallЙ
lstm_68/StatefulPartitionedCallStatefulPartitionedCallinputslstm_68_50187393lstm_68_50187395lstm_68_50187397*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_68_layer_call_and_return_conditional_losses_501863972!
lstm_68/StatefulPartitionedCallІ
gru_59/StatefulPartitionedCallStatefulPartitionedCallinputsgru_59_50187400gru_59_50187402gru_59_50187404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501867442 
gru_59/StatefulPartitionedCall
dropout_68/PartitionedCallPartitionedCall(lstm_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867912
dropout_68/PartitionedCall
dropout_69/PartitionedCallPartitionedCall'gru_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_69_layer_call_and_return_conditional_losses_501868212
dropout_69/PartitionedCallЩ
lstm_69/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0lstm_69_50187409lstm_69_50187411lstm_69_50187413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501871392!
lstm_69/StatefulPartitionedCallФ
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(lstm_69/StatefulPartitionedCall:output:0dense_327_50187416dense_327_50187418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_327_layer_call_and_return_conditional_losses_501871802#
!dense_327/StatefulPartitionedCallП
!dense_328/StatefulPartitionedCallStatefulPartitionedCall#dropout_69/PartitionedCall:output:0dense_328_50187421dense_328_50187423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_328_layer_call_and_return_conditional_losses_501872072#
!dense_328/StatefulPartitionedCallЙ
concatenate_9/PartitionedCallPartitionedCall*dense_327/StatefulPartitionedCall:output:0*dense_328/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_501872302
concatenate_9/PartitionedCallТ
!dense_329/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_329_50187427dense_329_50187429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_329_layer_call_and_return_conditional_losses_501872492#
!dense_329/StatefulPartitionedCallЯ
IdentityIdentity*dense_329/StatefulPartitionedCall:output:0"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall^gru_59/StatefulPartitionedCall ^lstm_68/StatefulPartitionedCall ^lstm_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2@
gru_59/StatefulPartitionedCallgru_59/StatefulPartitionedCall2B
lstm_68/StatefulPartitionedCalllstm_68/StatefulPartitionedCall2B
lstm_69/StatefulPartitionedCalllstm_69/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
"
щ
while_body_50185405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0 
while_gru_cell_59_50185427_0 
while_gru_cell_59_50185429_0 
while_gru_cell_59_50185431_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_59_50185427
while_gru_cell_59_50185429
while_gru_cell_59_50185431Ђ)while/gru_cell_59/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemМ
)while/gru_cell_59/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_59_50185427_0while_gru_cell_59_50185429_0while_gru_cell_59_50185431_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501850282+
)while/gru_cell_59/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_59/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_59/StatefulPartitionedCall:output:1*^while/gru_cell_59/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4":
while_gru_cell_59_50185427while_gru_cell_59_50185427_0":
while_gru_cell_59_50185429while_gru_cell_59_50185429_0":
while_gru_cell_59_50185431while_gru_cell_59_50185431_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2V
)while/gru_cell_59/StatefulPartitionedCall)while/gru_cell_59/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
йя
 
$__inference__traced_restore_50191352
file_prefix%
!assignvariableop_dense_327_kernel%
!assignvariableop_1_dense_327_bias'
#assignvariableop_2_dense_328_kernel%
!assignvariableop_3_dense_328_bias'
#assignvariableop_4_dense_329_kernel%
!assignvariableop_5_dense_329_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate3
/assignvariableop_11_lstm_68_lstm_cell_68_kernel=
9assignvariableop_12_lstm_68_lstm_cell_68_recurrent_kernel1
-assignvariableop_13_lstm_68_lstm_cell_68_bias1
-assignvariableop_14_gru_59_gru_cell_59_kernel;
7assignvariableop_15_gru_59_gru_cell_59_recurrent_kernel/
+assignvariableop_16_gru_59_gru_cell_59_bias3
/assignvariableop_17_lstm_69_lstm_cell_69_kernel=
9assignvariableop_18_lstm_69_lstm_cell_69_recurrent_kernel1
-assignvariableop_19_lstm_69_lstm_cell_69_bias
assignvariableop_20_total
assignvariableop_21_count
assignvariableop_22_total_1
assignvariableop_23_count_1
assignvariableop_24_total_2
assignvariableop_25_count_2/
+assignvariableop_26_adam_dense_327_kernel_m-
)assignvariableop_27_adam_dense_327_bias_m/
+assignvariableop_28_adam_dense_328_kernel_m-
)assignvariableop_29_adam_dense_328_bias_m/
+assignvariableop_30_adam_dense_329_kernel_m-
)assignvariableop_31_adam_dense_329_bias_m:
6assignvariableop_32_adam_lstm_68_lstm_cell_68_kernel_mD
@assignvariableop_33_adam_lstm_68_lstm_cell_68_recurrent_kernel_m8
4assignvariableop_34_adam_lstm_68_lstm_cell_68_bias_m8
4assignvariableop_35_adam_gru_59_gru_cell_59_kernel_mB
>assignvariableop_36_adam_gru_59_gru_cell_59_recurrent_kernel_m6
2assignvariableop_37_adam_gru_59_gru_cell_59_bias_m:
6assignvariableop_38_adam_lstm_69_lstm_cell_69_kernel_mD
@assignvariableop_39_adam_lstm_69_lstm_cell_69_recurrent_kernel_m8
4assignvariableop_40_adam_lstm_69_lstm_cell_69_bias_m/
+assignvariableop_41_adam_dense_327_kernel_v-
)assignvariableop_42_adam_dense_327_bias_v/
+assignvariableop_43_adam_dense_328_kernel_v-
)assignvariableop_44_adam_dense_328_bias_v/
+assignvariableop_45_adam_dense_329_kernel_v-
)assignvariableop_46_adam_dense_329_bias_v:
6assignvariableop_47_adam_lstm_68_lstm_cell_68_kernel_vD
@assignvariableop_48_adam_lstm_68_lstm_cell_68_recurrent_kernel_v8
4assignvariableop_49_adam_lstm_68_lstm_cell_68_bias_v8
4assignvariableop_50_adam_gru_59_gru_cell_59_kernel_vB
>assignvariableop_51_adam_gru_59_gru_cell_59_recurrent_kernel_v6
2assignvariableop_52_adam_gru_59_gru_cell_59_bias_v:
6assignvariableop_53_adam_lstm_69_lstm_cell_69_kernel_vD
@assignvariableop_54_adam_lstm_69_lstm_cell_69_recurrent_kernel_v8
4assignvariableop_55_adam_lstm_69_lstm_cell_69_bias_v
identity_57ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9М
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Ш
valueОBЛ9B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЫ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*њ
_output_shapesч
ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_327_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_327_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_328_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_328_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ј
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_329_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5І
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_329_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11З
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_68_lstm_cell_68_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12С
AssignVariableOp_12AssignVariableOp9assignvariableop_12_lstm_68_lstm_cell_68_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Е
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_68_lstm_cell_68_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Е
AssignVariableOp_14AssignVariableOp-assignvariableop_14_gru_59_gru_cell_59_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15П
AssignVariableOp_15AssignVariableOp7assignvariableop_15_gru_59_gru_cell_59_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Г
AssignVariableOp_16AssignVariableOp+assignvariableop_16_gru_59_gru_cell_59_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17З
AssignVariableOp_17AssignVariableOp/assignvariableop_17_lstm_69_lstm_cell_69_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18С
AssignVariableOp_18AssignVariableOp9assignvariableop_18_lstm_69_lstm_cell_69_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Е
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_69_lstm_cell_69_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ё
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ё
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѓ
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ѓ
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѓ
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Г
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_327_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Б
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_327_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Г
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_328_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Б
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_328_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Г
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_329_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Б
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_329_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32О
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_68_lstm_cell_68_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ш
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_68_lstm_cell_68_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34М
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_68_lstm_cell_68_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35М
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_gru_59_gru_cell_59_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ц
AssignVariableOp_36AssignVariableOp>assignvariableop_36_adam_gru_59_gru_cell_59_recurrent_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37К
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_gru_59_gru_cell_59_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38О
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_lstm_69_lstm_cell_69_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ш
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_lstm_69_lstm_cell_69_recurrent_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40М
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_lstm_69_lstm_cell_69_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_327_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Б
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_327_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Г
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_328_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Б
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_328_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Г
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_329_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_329_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47О
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_lstm_68_lstm_cell_68_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ш
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_lstm_68_lstm_cell_68_recurrent_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49М
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_lstm_68_lstm_cell_68_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50М
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_gru_59_gru_cell_59_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ц
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_gru_59_gru_cell_59_recurrent_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52К
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_gru_59_gru_cell_59_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53О
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_lstm_69_lstm_cell_69_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ш
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_lstm_69_lstm_cell_69_recurrent_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55М
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_lstm_69_lstm_cell_69_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_559
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЎ

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_56Ё

Identity_57IdentityIdentity_56:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_57"#
identity_57Identity_57:output:0*ї
_input_shapesх
т: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Щ[
є
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189038

inputs/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50188953*
condR
while_cond_50188952*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќD
ф
E__inference_lstm_68_layer_call_and_return_conditional_losses_50184775

inputs
lstm_cell_68_50184693
lstm_cell_68_50184695
lstm_cell_68_50184697
identityЂ$lstm_cell_68/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_68/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_68_50184693lstm_cell_68_50184695lstm_cell_68_50184697*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501843792&
$lstm_cell_68/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_68_50184693lstm_cell_68_50184695lstm_cell_68_50184697*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50184706*
condR
while_cond_50184705*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0%^lstm_cell_68/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2L
$lstm_cell_68/StatefulPartitionedCall$lstm_cell_68/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ
f
H__inference_dropout_68_layer_call_and_return_conditional_losses_50186791

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Е
Э
while_cond_50190315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50190315___redundant_placeholder06
2while_while_cond_50190315___redundant_placeholder16
2while_while_cond_50190315___redundant_placeholder26
2while_while_cond_50190315___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ќD
ф
E__inference_lstm_68_layer_call_and_return_conditional_losses_50184907

inputs
lstm_cell_68_50184825
lstm_cell_68_50184827
lstm_cell_68_50184829
identityЂ$lstm_cell_68/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_68/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_68_50184825lstm_cell_68_50184827lstm_cell_68_50184829*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501844122&
$lstm_cell_68/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_68_50184825lstm_cell_68_50184827lstm_cell_68_50184829*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50184838*
condR
while_cond_50184837*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0%^lstm_cell_68/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2L
$lstm_cell_68/StatefulPartitionedCall$lstm_cell_68/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё	
р
G__inference_dense_328_layer_call_and_return_conditional_losses_50187207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Е
Э
while_cond_50187053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50187053___redundant_placeholder06
2while_while_cond_50187053___redundant_placeholder16
2while_while_cond_50187053___redundant_placeholder26
2while_while_cond_50187053___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
јD
ф
E__inference_lstm_69_layer_call_and_return_conditional_losses_50186079

inputs
lstm_cell_69_50185997
lstm_cell_69_50185999
lstm_cell_69_50186001
identityЂ$lstm_cell_69/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_69/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_69_50185997lstm_cell_69_50185999lstm_cell_69_50186001*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855842&
$lstm_cell_69/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_69_50185997lstm_cell_69_50185999lstm_cell_69_50186001*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50186010*
condR
while_cond_50186009*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_69/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2L
$lstm_cell_69/StatefulPartitionedCall$lstm_cell_69/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
	
р
G__inference_dense_329_layer_call_and_return_conditional_losses_50187249

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
O


lstm_69_while_body_50187899,
(lstm_69_while_lstm_69_while_loop_counter2
.lstm_69_while_lstm_69_while_maximum_iterations
lstm_69_while_placeholder
lstm_69_while_placeholder_1
lstm_69_while_placeholder_2
lstm_69_while_placeholder_3+
'lstm_69_while_lstm_69_strided_slice_1_0g
clstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0A
=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0@
<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0
lstm_69_while_identity
lstm_69_while_identity_1
lstm_69_while_identity_2
lstm_69_while_identity_3
lstm_69_while_identity_4
lstm_69_while_identity_5)
%lstm_69_while_lstm_69_strided_slice_1e
alstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor=
9lstm_69_while_lstm_cell_69_matmul_readvariableop_resource?
;lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource>
:lstm_69_while_lstm_cell_69_biasadd_readvariableop_resourceЂ1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpЂ0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpЂ2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpг
?lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2A
?lstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_69/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0lstm_69_while_placeholderHlstm_69/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype023
1lstm_69/while/TensorArrayV2Read/TensorListGetItemс
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype022
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOpї
!lstm_69/while/lstm_cell_69/MatMulMatMul8lstm_69/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!lstm_69/while/lstm_cell_69/MatMulч
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype024
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOpр
#lstm_69/while/lstm_cell_69/MatMul_1MatMullstm_69_while_placeholder_2:lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#lstm_69/while/lstm_cell_69/MatMul_1и
lstm_69/while/lstm_cell_69/addAddV2+lstm_69/while/lstm_cell_69/MatMul:product:0-lstm_69/while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
lstm_69/while/lstm_cell_69/addр
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOpх
"lstm_69/while/lstm_cell_69/BiasAddBiasAdd"lstm_69/while/lstm_cell_69/add:z:09lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"lstm_69/while/lstm_cell_69/BiasAdd
 lstm_69/while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_69/while/lstm_cell_69/Const
*lstm_69/while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_69/while/lstm_cell_69/split/split_dimЋ
 lstm_69/while/lstm_cell_69/splitSplit3lstm_69/while/lstm_cell_69/split/split_dim:output:0+lstm_69/while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 lstm_69/while/lstm_cell_69/splitА
"lstm_69/while/lstm_cell_69/SigmoidSigmoid)lstm_69/while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"lstm_69/while/lstm_cell_69/SigmoidД
$lstm_69/while/lstm_cell_69/Sigmoid_1Sigmoid)lstm_69/while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_69/while/lstm_cell_69/Sigmoid_1Р
lstm_69/while/lstm_cell_69/mulMul(lstm_69/while/lstm_cell_69/Sigmoid_1:y:0lstm_69_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/while/lstm_cell_69/mulЇ
lstm_69/while/lstm_cell_69/ReluRelu)lstm_69/while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
lstm_69/while/lstm_cell_69/Reluд
 lstm_69/while/lstm_cell_69/mul_1Mul&lstm_69/while/lstm_cell_69/Sigmoid:y:0-lstm_69/while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/mul_1Щ
 lstm_69/while/lstm_cell_69/add_1AddV2"lstm_69/while/lstm_cell_69/mul:z:0$lstm_69/while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/add_1Д
$lstm_69/while/lstm_cell_69/Sigmoid_2Sigmoid)lstm_69/while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_69/while/lstm_cell_69/Sigmoid_2І
!lstm_69/while/lstm_cell_69/Relu_1Relu$lstm_69/while/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!lstm_69/while/lstm_cell_69/Relu_1и
 lstm_69/while/lstm_cell_69/mul_2Mul(lstm_69/while/lstm_cell_69/Sigmoid_2:y:0/lstm_69/while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_69/while/lstm_cell_69/mul_2
2lstm_69/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_69_while_placeholder_1lstm_69_while_placeholder$lstm_69/while/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_69/while/TensorArrayV2Write/TensorListSetIteml
lstm_69/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/while/add/y
lstm_69/while/addAddV2lstm_69_while_placeholderlstm_69/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_69/while/addp
lstm_69/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/while/add_1/y
lstm_69/while/add_1AddV2(lstm_69_while_lstm_69_while_loop_counterlstm_69/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_69/while/add_1
lstm_69/while/IdentityIdentitylstm_69/while/add_1:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity­
lstm_69/while/Identity_1Identity.lstm_69_while_lstm_69_while_maximum_iterations2^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_1
lstm_69/while/Identity_2Identitylstm_69/while/add:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_2С
lstm_69/while/Identity_3IdentityBlstm_69/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_69/while/Identity_3Д
lstm_69/while/Identity_4Identity$lstm_69/while/lstm_cell_69/mul_2:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/while/Identity_4Д
lstm_69/while/Identity_5Identity$lstm_69/while/lstm_cell_69/add_1:z:02^lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1^lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp3^lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/while/Identity_5"9
lstm_69_while_identitylstm_69/while/Identity:output:0"=
lstm_69_while_identity_1!lstm_69/while/Identity_1:output:0"=
lstm_69_while_identity_2!lstm_69/while/Identity_2:output:0"=
lstm_69_while_identity_3!lstm_69/while/Identity_3:output:0"=
lstm_69_while_identity_4!lstm_69/while/Identity_4:output:0"=
lstm_69_while_identity_5!lstm_69/while/Identity_5:output:0"P
%lstm_69_while_lstm_69_strided_slice_1'lstm_69_while_lstm_69_strided_slice_1_0"z
:lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource<lstm_69_while_lstm_cell_69_biasadd_readvariableop_resource_0"|
;lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource=lstm_69_while_lstm_cell_69_matmul_1_readvariableop_resource_0"x
9lstm_69_while_lstm_cell_69_matmul_readvariableop_resource;lstm_69_while_lstm_cell_69_matmul_readvariableop_resource_0"Ш
alstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensorclstm_69_while_tensorarrayv2read_tensorlistgetitem_lstm_69_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2f
1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp1lstm_69/while/lstm_cell_69/BiasAdd/ReadVariableOp2d
0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp0lstm_69/while/lstm_cell_69/MatMul/ReadVariableOp2h
2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp2lstm_69/while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_50186900
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50186900___redundant_placeholder06
2while_while_cond_50186900___redundant_placeholder16
2while_while_cond_50186900___redundant_placeholder26
2while_while_cond_50186900___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
н
f
-__inference_dropout_68_layer_call_fn_50189235

inputs
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_68_layer_call_and_return_conditional_losses_501867862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
њG
А
while_body_50186654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
C

while_body_50190316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ё	
р
G__inference_dense_328_layer_call_and_return_conditional_losses_50190634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
х	
А
.__inference_gru_cell_59_layer_call_fn_50190869

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501849882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
Э[
і
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190226
inputs_0/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50190141*
condR
while_cond_50190140*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
Е
Э
while_cond_50184705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50184705___redundant_placeholder06
2while_while_cond_50184705___redundant_placeholder16
2while_while_cond_50184705___redundant_placeholder26
2while_while_cond_50184705___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
ь

E__inference_model_9_layer_call_and_return_conditional_losses_50188006

inputs7
3lstm_68_lstm_cell_68_matmul_readvariableop_resource9
5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource8
4lstm_68_lstm_cell_68_biasadd_readvariableop_resource.
*gru_59_gru_cell_59_readvariableop_resource5
1gru_59_gru_cell_59_matmul_readvariableop_resource7
3gru_59_gru_cell_59_matmul_1_readvariableop_resource7
3lstm_69_lstm_cell_69_matmul_readvariableop_resource9
5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource8
4lstm_69_lstm_cell_69_biasadd_readvariableop_resource,
(dense_327_matmul_readvariableop_resource-
)dense_327_biasadd_readvariableop_resource,
(dense_328_matmul_readvariableop_resource-
)dense_328_biasadd_readvariableop_resource,
(dense_329_matmul_readvariableop_resource-
)dense_329_biasadd_readvariableop_resource
identityЂ dense_327/BiasAdd/ReadVariableOpЂdense_327/MatMul/ReadVariableOpЂ dense_328/BiasAdd/ReadVariableOpЂdense_328/MatMul/ReadVariableOpЂ dense_329/BiasAdd/ReadVariableOpЂdense_329/MatMul/ReadVariableOpЂ(gru_59/gru_cell_59/MatMul/ReadVariableOpЂ*gru_59/gru_cell_59/MatMul_1/ReadVariableOpЂ!gru_59/gru_cell_59/ReadVariableOpЂgru_59/whileЂ+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpЂ*lstm_68/lstm_cell_68/MatMul/ReadVariableOpЂ,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpЂlstm_68/whileЂ+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpЂ*lstm_69/lstm_cell_69/MatMul/ReadVariableOpЂ,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpЂlstm_69/whileT
lstm_68/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_68/Shape
lstm_68/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice/stack
lstm_68/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_68/strided_slice/stack_1
lstm_68/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_68/strided_slice/stack_2
lstm_68/strided_sliceStridedSlicelstm_68/Shape:output:0$lstm_68/strided_slice/stack:output:0&lstm_68/strided_slice/stack_1:output:0&lstm_68/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_68/strided_slicel
lstm_68/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros/mul/y
lstm_68/zeros/mulMullstm_68/strided_slice:output:0lstm_68/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros/mulo
lstm_68/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_68/zeros/Less/y
lstm_68/zeros/LessLesslstm_68/zeros/mul:z:0lstm_68/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros/Lessr
lstm_68/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros/packed/1Ѓ
lstm_68/zeros/packedPacklstm_68/strided_slice:output:0lstm_68/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_68/zeros/packedo
lstm_68/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/zeros/Const
lstm_68/zerosFilllstm_68/zeros/packed:output:0lstm_68/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/zerosp
lstm_68/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros_1/mul/y
lstm_68/zeros_1/mulMullstm_68/strided_slice:output:0lstm_68/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros_1/muls
lstm_68/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_68/zeros_1/Less/y
lstm_68/zeros_1/LessLesslstm_68/zeros_1/mul:z:0lstm_68/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_68/zeros_1/Lessv
lstm_68/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_68/zeros_1/packed/1Љ
lstm_68/zeros_1/packedPacklstm_68/strided_slice:output:0!lstm_68/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_68/zeros_1/packeds
lstm_68/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/zeros_1/Const
lstm_68/zeros_1Filllstm_68/zeros_1/packed:output:0lstm_68/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/zeros_1
lstm_68/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_68/transpose/perm
lstm_68/transpose	Transposeinputslstm_68/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
lstm_68/transposeg
lstm_68/Shape_1Shapelstm_68/transpose:y:0*
T0*
_output_shapes
:2
lstm_68/Shape_1
lstm_68/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice_1/stack
lstm_68/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_1/stack_1
lstm_68/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_1/stack_2
lstm_68/strided_slice_1StridedSlicelstm_68/Shape_1:output:0&lstm_68/strided_slice_1/stack:output:0(lstm_68/strided_slice_1/stack_1:output:0(lstm_68/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_68/strided_slice_1
#lstm_68/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_68/TensorArrayV2/element_shapeв
lstm_68/TensorArrayV2TensorListReserve,lstm_68/TensorArrayV2/element_shape:output:0 lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_68/TensorArrayV2Я
=lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_68/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_68/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_68/transpose:y:0Flstm_68/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_68/TensorArrayUnstack/TensorListFromTensor
lstm_68/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_68/strided_slice_2/stack
lstm_68/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_2/stack_1
lstm_68/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_2/stack_2Ќ
lstm_68/strided_slice_2StridedSlicelstm_68/transpose:y:0&lstm_68/strided_slice_2/stack:output:0(lstm_68/strided_slice_2/stack_1:output:0(lstm_68/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_68/strided_slice_2Э
*lstm_68/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3lstm_68_lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02,
*lstm_68/lstm_cell_68/MatMul/ReadVariableOpЭ
lstm_68/lstm_cell_68/MatMulMatMul lstm_68/strided_slice_2:output:02lstm_68/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/MatMulг
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02.
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOpЩ
lstm_68/lstm_cell_68/MatMul_1MatMullstm_68/zeros:output:04lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/MatMul_1Р
lstm_68/lstm_cell_68/addAddV2%lstm_68/lstm_cell_68/MatMul:product:0'lstm_68/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/addЬ
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02-
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOpЭ
lstm_68/lstm_cell_68/BiasAddBiasAddlstm_68/lstm_cell_68/add:z:03lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_68/lstm_cell_68/BiasAddz
lstm_68/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_68/lstm_cell_68/Const
$lstm_68/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_68/lstm_cell_68/split/split_dim
lstm_68/lstm_cell_68/splitSplit-lstm_68/lstm_cell_68/split/split_dim:output:0%lstm_68/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_68/lstm_cell_68/split
lstm_68/lstm_cell_68/SigmoidSigmoid#lstm_68/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/SigmoidЂ
lstm_68/lstm_cell_68/Sigmoid_1Sigmoid#lstm_68/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/lstm_cell_68/Sigmoid_1Ћ
lstm_68/lstm_cell_68/mulMul"lstm_68/lstm_cell_68/Sigmoid_1:y:0lstm_68/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul
lstm_68/lstm_cell_68/ReluRelu#lstm_68/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/ReluМ
lstm_68/lstm_cell_68/mul_1Mul lstm_68/lstm_cell_68/Sigmoid:y:0'lstm_68/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul_1Б
lstm_68/lstm_cell_68/add_1AddV2lstm_68/lstm_cell_68/mul:z:0lstm_68/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/add_1Ђ
lstm_68/lstm_cell_68/Sigmoid_2Sigmoid#lstm_68/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_68/lstm_cell_68/Sigmoid_2
lstm_68/lstm_cell_68/Relu_1Relulstm_68/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/Relu_1Р
lstm_68/lstm_cell_68/mul_2Mul"lstm_68/lstm_cell_68/Sigmoid_2:y:0)lstm_68/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_68/lstm_cell_68/mul_2
%lstm_68/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2'
%lstm_68/TensorArrayV2_1/element_shapeи
lstm_68/TensorArrayV2_1TensorListReserve.lstm_68/TensorArrayV2_1/element_shape:output:0 lstm_68/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_68/TensorArrayV2_1^
lstm_68/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_68/time
 lstm_68/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_68/while/maximum_iterationsz
lstm_68/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_68/while/loop_counterъ
lstm_68/whileWhile#lstm_68/while/loop_counter:output:0)lstm_68/while/maximum_iterations:output:0lstm_68/time:output:0 lstm_68/TensorArrayV2_1:handle:0lstm_68/zeros:output:0lstm_68/zeros_1:output:0 lstm_68/strided_slice_1:output:0?lstm_68/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_68_lstm_cell_68_matmul_readvariableop_resource5lstm_68_lstm_cell_68_matmul_1_readvariableop_resource4lstm_68_lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_68_while_body_50187579*'
condR
lstm_68_while_cond_50187578*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
lstm_68/whileХ
8lstm_68/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2:
8lstm_68/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_68/TensorArrayV2Stack/TensorListStackTensorListStacklstm_68/while:output:3Alstm_68/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02,
*lstm_68/TensorArrayV2Stack/TensorListStack
lstm_68/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_68/strided_slice_3/stack
lstm_68/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_68/strided_slice_3/stack_1
lstm_68/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_68/strided_slice_3/stack_2Ъ
lstm_68/strided_slice_3StridedSlice3lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_68/strided_slice_3/stack:output:0(lstm_68/strided_slice_3/stack_1:output:0(lstm_68/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_68/strided_slice_3
lstm_68/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_68/transpose_1/permЮ
lstm_68/transpose_1	Transpose3lstm_68/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_68/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_68/transpose_1v
lstm_68/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_68/runtimeR
gru_59/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_59/Shape
gru_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice/stack
gru_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_59/strided_slice/stack_1
gru_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_59/strided_slice/stack_2
gru_59/strided_sliceStridedSlicegru_59/Shape:output:0#gru_59/strided_slice/stack:output:0%gru_59/strided_slice/stack_1:output:0%gru_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_59/strided_slicej
gru_59/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_59/zeros/mul/y
gru_59/zeros/mulMulgru_59/strided_slice:output:0gru_59/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_59/zeros/mulm
gru_59/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru_59/zeros/Less/y
gru_59/zeros/LessLessgru_59/zeros/mul:z:0gru_59/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_59/zeros/Lessp
gru_59/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_59/zeros/packed/1
gru_59/zeros/packedPackgru_59/strided_slice:output:0gru_59/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_59/zeros/packedm
gru_59/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_59/zeros/Const
gru_59/zerosFillgru_59/zeros/packed:output:0gru_59/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/zeros
gru_59/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_59/transpose/perm
gru_59/transpose	Transposeinputsgru_59/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru_59/transposed
gru_59/Shape_1Shapegru_59/transpose:y:0*
T0*
_output_shapes
:2
gru_59/Shape_1
gru_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice_1/stack
gru_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_1/stack_1
gru_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_1/stack_2
gru_59/strided_slice_1StridedSlicegru_59/Shape_1:output:0%gru_59/strided_slice_1/stack:output:0'gru_59/strided_slice_1/stack_1:output:0'gru_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_59/strided_slice_1
"gru_59/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_59/TensorArrayV2/element_shapeЮ
gru_59/TensorArrayV2TensorListReserve+gru_59/TensorArrayV2/element_shape:output:0gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_59/TensorArrayV2Э
<gru_59/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2>
<gru_59/TensorArrayUnstack/TensorListFromTensor/element_shape
.gru_59/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_59/transpose:y:0Egru_59/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_59/TensorArrayUnstack/TensorListFromTensor
gru_59/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_59/strided_slice_2/stack
gru_59/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_2/stack_1
gru_59/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_2/stack_2І
gru_59/strided_slice_2StridedSlicegru_59/transpose:y:0%gru_59/strided_slice_2/stack:output:0'gru_59/strided_slice_2/stack_1:output:0'gru_59/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
gru_59/strided_slice_2В
!gru_59/gru_cell_59/ReadVariableOpReadVariableOp*gru_59_gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_59/gru_cell_59/ReadVariableOpЅ
gru_59/gru_cell_59/unstackUnpack)gru_59/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_59/gru_cell_59/unstackЧ
(gru_59/gru_cell_59/MatMul/ReadVariableOpReadVariableOp1gru_59_gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(gru_59/gru_cell_59/MatMul/ReadVariableOpЦ
gru_59/gru_cell_59/MatMulMatMulgru_59/strided_slice_2:output:00gru_59/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/MatMulР
gru_59/gru_cell_59/BiasAddBiasAdd#gru_59/gru_cell_59/MatMul:product:0#gru_59/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/BiasAddv
gru_59/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/gru_cell_59/Const
"gru_59/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_59/gru_cell_59/split/split_dimј
gru_59/gru_cell_59/splitSplit+gru_59/gru_cell_59/split/split_dim:output:0#gru_59/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_59/gru_cell_59/splitЭ
*gru_59/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp3gru_59_gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02,
*gru_59/gru_cell_59/MatMul_1/ReadVariableOpТ
gru_59/gru_cell_59/MatMul_1MatMulgru_59/zeros:output:02gru_59/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/MatMul_1Ц
gru_59/gru_cell_59/BiasAdd_1BiasAdd%gru_59/gru_cell_59/MatMul_1:product:0#gru_59/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_59/gru_cell_59/BiasAdd_1
gru_59/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_59/gru_cell_59/Const_1
$gru_59/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$gru_59/gru_cell_59/split_1/split_dimВ
gru_59/gru_cell_59/split_1SplitV%gru_59/gru_cell_59/BiasAdd_1:output:0#gru_59/gru_cell_59/Const_1:output:0-gru_59/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_59/gru_cell_59/split_1Г
gru_59/gru_cell_59/addAddV2!gru_59/gru_cell_59/split:output:0#gru_59/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add
gru_59/gru_cell_59/SigmoidSigmoidgru_59/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/SigmoidЗ
gru_59/gru_cell_59/add_1AddV2!gru_59/gru_cell_59/split:output:1#gru_59/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_1
gru_59/gru_cell_59/Sigmoid_1Sigmoidgru_59/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/Sigmoid_1А
gru_59/gru_cell_59/mulMul gru_59/gru_cell_59/Sigmoid_1:y:0#gru_59/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mulЎ
gru_59/gru_cell_59/add_2AddV2!gru_59/gru_cell_59/split:output:2gru_59/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_2
gru_59/gru_cell_59/ReluRelugru_59/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/ReluЄ
gru_59/gru_cell_59/mul_1Mulgru_59/gru_cell_59/Sigmoid:y:0gru_59/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mul_1y
gru_59/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_59/gru_cell_59/sub/xЌ
gru_59/gru_cell_59/subSub!gru_59/gru_cell_59/sub/x:output:0gru_59/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/subА
gru_59/gru_cell_59/mul_2Mulgru_59/gru_cell_59/sub:z:0%gru_59/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/mul_2Ћ
gru_59/gru_cell_59/add_3AddV2gru_59/gru_cell_59/mul_1:z:0gru_59/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/gru_cell_59/add_3
$gru_59/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2&
$gru_59/TensorArrayV2_1/element_shapeд
gru_59/TensorArrayV2_1TensorListReserve-gru_59/TensorArrayV2_1/element_shape:output:0gru_59/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_59/TensorArrayV2_1\
gru_59/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_59/time
gru_59/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
gru_59/while/maximum_iterationsx
gru_59/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_59/while/loop_counter
gru_59/whileWhile"gru_59/while/loop_counter:output:0(gru_59/while/maximum_iterations:output:0gru_59/time:output:0gru_59/TensorArrayV2_1:handle:0gru_59/zeros:output:0gru_59/strided_slice_1:output:0>gru_59/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_59_gru_cell_59_readvariableop_resource1gru_59_gru_cell_59_matmul_readvariableop_resource3gru_59_gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*&
bodyR
gru_59_while_body_50187729*&
condR
gru_59_while_cond_50187728*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
gru_59/whileУ
7gru_59/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   29
7gru_59/TensorArrayV2Stack/TensorListStack/element_shape
)gru_59/TensorArrayV2Stack/TensorListStackTensorListStackgru_59/while:output:3@gru_59/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02+
)gru_59/TensorArrayV2Stack/TensorListStack
gru_59/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
gru_59/strided_slice_3/stack
gru_59/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_59/strided_slice_3/stack_1
gru_59/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_59/strided_slice_3/stack_2Ф
gru_59/strided_slice_3StridedSlice2gru_59/TensorArrayV2Stack/TensorListStack:tensor:0%gru_59/strided_slice_3/stack:output:0'gru_59/strided_slice_3/stack_1:output:0'gru_59/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
gru_59/strided_slice_3
gru_59/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_59/transpose_1/permЪ
gru_59/transpose_1	Transpose2gru_59/TensorArrayV2Stack/TensorListStack:tensor:0 gru_59/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
gru_59/transpose_1t
gru_59/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_59/runtimey
dropout_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout_68/dropout/ConstВ
dropout_68/dropout/MulMullstm_68/transpose_1:y:0!dropout_68/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_68/dropout/Mul{
dropout_68/dropout/ShapeShapelstm_68/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_68/dropout/Shapeт
/dropout_68/dropout/random_uniform/RandomUniformRandomUniform!dropout_68/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype021
/dropout_68/dropout/random_uniform/RandomUniform
!dropout_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2#
!dropout_68/dropout/GreaterEqual/yї
dropout_68/dropout/GreaterEqualGreaterEqual8dropout_68/dropout/random_uniform/RandomUniform:output:0*dropout_68/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2!
dropout_68/dropout/GreaterEqual­
dropout_68/dropout/CastCast#dropout_68/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_68/dropout/CastГ
dropout_68/dropout/Mul_1Muldropout_68/dropout/Mul:z:0dropout_68/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_68/dropout/Mul_1y
dropout_69/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_69/dropout/Const­
dropout_69/dropout/MulMulgru_59/strided_slice_3:output:0!dropout_69/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_69/dropout/Mul
dropout_69/dropout/ShapeShapegru_59/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_69/dropout/Shapeе
/dropout_69/dropout/random_uniform/RandomUniformRandomUniform!dropout_69/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype021
/dropout_69/dropout/random_uniform/RandomUniform
!dropout_69/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_69/dropout/GreaterEqual/yъ
dropout_69/dropout/GreaterEqualGreaterEqual8dropout_69/dropout/random_uniform/RandomUniform:output:0*dropout_69/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
dropout_69/dropout/GreaterEqual 
dropout_69/dropout/CastCast#dropout_69/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout_69/dropout/CastІ
dropout_69/dropout/Mul_1Muldropout_69/dropout/Mul:z:0dropout_69/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_69/dropout/Mul_1j
lstm_69/ShapeShapedropout_68/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_69/Shape
lstm_69/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice/stack
lstm_69/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_69/strided_slice/stack_1
lstm_69/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_69/strided_slice/stack_2
lstm_69/strided_sliceStridedSlicelstm_69/Shape:output:0$lstm_69/strided_slice/stack:output:0&lstm_69/strided_slice/stack_1:output:0&lstm_69/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_69/strided_slicel
lstm_69/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros/mul/y
lstm_69/zeros/mulMullstm_69/strided_slice:output:0lstm_69/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros/mulo
lstm_69/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_69/zeros/Less/y
lstm_69/zeros/LessLesslstm_69/zeros/mul:z:0lstm_69/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros/Lessr
lstm_69/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros/packed/1Ѓ
lstm_69/zeros/packedPacklstm_69/strided_slice:output:0lstm_69/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_69/zeros/packedo
lstm_69/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/zeros/Const
lstm_69/zerosFilllstm_69/zeros/packed:output:0lstm_69/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/zerosp
lstm_69/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros_1/mul/y
lstm_69/zeros_1/mulMullstm_69/strided_slice:output:0lstm_69/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros_1/muls
lstm_69/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_69/zeros_1/Less/y
lstm_69/zeros_1/LessLesslstm_69/zeros_1/mul:z:0lstm_69/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_69/zeros_1/Lessv
lstm_69/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_69/zeros_1/packed/1Љ
lstm_69/zeros_1/packedPacklstm_69/strided_slice:output:0!lstm_69/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_69/zeros_1/packeds
lstm_69/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/zeros_1/Const
lstm_69/zeros_1Filllstm_69/zeros_1/packed:output:0lstm_69/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/zeros_1
lstm_69/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_69/transpose/permБ
lstm_69/transpose	Transposedropout_68/dropout/Mul_1:z:0lstm_69/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_69/transposeg
lstm_69/Shape_1Shapelstm_69/transpose:y:0*
T0*
_output_shapes
:2
lstm_69/Shape_1
lstm_69/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice_1/stack
lstm_69/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_1/stack_1
lstm_69/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_1/stack_2
lstm_69/strided_slice_1StridedSlicelstm_69/Shape_1:output:0&lstm_69/strided_slice_1/stack:output:0(lstm_69/strided_slice_1/stack_1:output:0(lstm_69/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_69/strided_slice_1
#lstm_69/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_69/TensorArrayV2/element_shapeв
lstm_69/TensorArrayV2TensorListReserve,lstm_69/TensorArrayV2/element_shape:output:0 lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_69/TensorArrayV2Я
=lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2?
=lstm_69/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_69/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_69/transpose:y:0Flstm_69/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_69/TensorArrayUnstack/TensorListFromTensor
lstm_69/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_69/strided_slice_2/stack
lstm_69/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_2/stack_1
lstm_69/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_2/stack_2Ќ
lstm_69/strided_slice_2StridedSlicelstm_69/transpose:y:0&lstm_69/strided_slice_2/stack:output:0(lstm_69/strided_slice_2/stack_1:output:0(lstm_69/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_69/strided_slice_2Э
*lstm_69/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3lstm_69_lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02,
*lstm_69/lstm_cell_69/MatMul/ReadVariableOpЭ
lstm_69/lstm_cell_69/MatMulMatMul lstm_69/strided_slice_2:output:02lstm_69/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/MatMulг
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02.
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOpЩ
lstm_69/lstm_cell_69/MatMul_1MatMullstm_69/zeros:output:04lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/MatMul_1Р
lstm_69/lstm_cell_69/addAddV2%lstm_69/lstm_cell_69/MatMul:product:0'lstm_69/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/addЬ
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOpЭ
lstm_69/lstm_cell_69/BiasAddBiasAddlstm_69/lstm_cell_69/add:z:03lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_69/lstm_cell_69/BiasAddz
lstm_69/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_69/lstm_cell_69/Const
$lstm_69/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_69/lstm_cell_69/split/split_dim
lstm_69/lstm_cell_69/splitSplit-lstm_69/lstm_cell_69/split/split_dim:output:0%lstm_69/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_69/lstm_cell_69/split
lstm_69/lstm_cell_69/SigmoidSigmoid#lstm_69/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/SigmoidЂ
lstm_69/lstm_cell_69/Sigmoid_1Sigmoid#lstm_69/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/lstm_cell_69/Sigmoid_1Ћ
lstm_69/lstm_cell_69/mulMul"lstm_69/lstm_cell_69/Sigmoid_1:y:0lstm_69/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul
lstm_69/lstm_cell_69/ReluRelu#lstm_69/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/ReluМ
lstm_69/lstm_cell_69/mul_1Mul lstm_69/lstm_cell_69/Sigmoid:y:0'lstm_69/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul_1Б
lstm_69/lstm_cell_69/add_1AddV2lstm_69/lstm_cell_69/mul:z:0lstm_69/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/add_1Ђ
lstm_69/lstm_cell_69/Sigmoid_2Sigmoid#lstm_69/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_69/lstm_cell_69/Sigmoid_2
lstm_69/lstm_cell_69/Relu_1Relulstm_69/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/Relu_1Р
lstm_69/lstm_cell_69/mul_2Mul"lstm_69/lstm_cell_69/Sigmoid_2:y:0)lstm_69/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_69/lstm_cell_69/mul_2
%lstm_69/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2'
%lstm_69/TensorArrayV2_1/element_shapeи
lstm_69/TensorArrayV2_1TensorListReserve.lstm_69/TensorArrayV2_1/element_shape:output:0 lstm_69/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_69/TensorArrayV2_1^
lstm_69/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_69/time
 lstm_69/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_69/while/maximum_iterationsz
lstm_69/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_69/while/loop_counterъ
lstm_69/whileWhile#lstm_69/while/loop_counter:output:0)lstm_69/while/maximum_iterations:output:0lstm_69/time:output:0 lstm_69/TensorArrayV2_1:handle:0lstm_69/zeros:output:0lstm_69/zeros_1:output:0 lstm_69/strided_slice_1:output:0?lstm_69/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_69_lstm_cell_69_matmul_readvariableop_resource5lstm_69_lstm_cell_69_matmul_1_readvariableop_resource4lstm_69_lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_69_while_body_50187899*'
condR
lstm_69_while_cond_50187898*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
lstm_69/whileХ
8lstm_69/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2:
8lstm_69/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_69/TensorArrayV2Stack/TensorListStackTensorListStacklstm_69/while:output:3Alstm_69/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02,
*lstm_69/TensorArrayV2Stack/TensorListStack
lstm_69/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_69/strided_slice_3/stack
lstm_69/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_69/strided_slice_3/stack_1
lstm_69/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_69/strided_slice_3/stack_2Ъ
lstm_69/strided_slice_3StridedSlice3lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_69/strided_slice_3/stack:output:0(lstm_69/strided_slice_3/stack_1:output:0(lstm_69/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
lstm_69/strided_slice_3
lstm_69/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_69/transpose_1/permЮ
lstm_69/transpose_1	Transpose3lstm_69/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_69/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
lstm_69/transpose_1v
lstm_69/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_69/runtimeЋ
dense_327/MatMul/ReadVariableOpReadVariableOp(dense_327_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02!
dense_327/MatMul/ReadVariableOpЋ
dense_327/MatMulMatMul lstm_69/strided_slice_3:output:0'dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/MatMulЊ
 dense_327/BiasAdd/ReadVariableOpReadVariableOp)dense_327_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_327/BiasAdd/ReadVariableOpЉ
dense_327/BiasAddBiasAdddense_327/MatMul:product:0(dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/BiasAddv
dense_327/ReluReludense_327/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_327/ReluЋ
dense_328/MatMul/ReadVariableOpReadVariableOp(dense_328_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02!
dense_328/MatMul/ReadVariableOpЇ
dense_328/MatMulMatMuldropout_69/dropout/Mul_1:z:0'dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/MatMulЊ
 dense_328/BiasAdd/ReadVariableOpReadVariableOp)dense_328_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_328/BiasAdd/ReadVariableOpЉ
dense_328/BiasAddBiasAdddense_328/MatMul:product:0(dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/BiasAddv
dense_328/ReluReludense_328/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_328/Relux
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axisг
concatenate_9/concatConcatV2dense_327/Relu:activations:0dense_328/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatenate_9/concatЋ
dense_329/MatMul/ReadVariableOpReadVariableOp(dense_329_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02!
dense_329/MatMul/ReadVariableOpЈ
dense_329/MatMulMatMulconcatenate_9/concat:output:0'dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_329/MatMulЊ
 dense_329/BiasAdd/ReadVariableOpReadVariableOp)dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_329/BiasAdd/ReadVariableOpЉ
dense_329/BiasAddBiasAdddense_329/MatMul:product:0(dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_329/BiasAddќ
IdentityIdentitydense_329/BiasAdd:output:0!^dense_327/BiasAdd/ReadVariableOp ^dense_327/MatMul/ReadVariableOp!^dense_328/BiasAdd/ReadVariableOp ^dense_328/MatMul/ReadVariableOp!^dense_329/BiasAdd/ReadVariableOp ^dense_329/MatMul/ReadVariableOp)^gru_59/gru_cell_59/MatMul/ReadVariableOp+^gru_59/gru_cell_59/MatMul_1/ReadVariableOp"^gru_59/gru_cell_59/ReadVariableOp^gru_59/while,^lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp+^lstm_68/lstm_cell_68/MatMul/ReadVariableOp-^lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp^lstm_68/while,^lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp+^lstm_69/lstm_cell_69/MatMul/ReadVariableOp-^lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp^lstm_69/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_327/BiasAdd/ReadVariableOp dense_327/BiasAdd/ReadVariableOp2B
dense_327/MatMul/ReadVariableOpdense_327/MatMul/ReadVariableOp2D
 dense_328/BiasAdd/ReadVariableOp dense_328/BiasAdd/ReadVariableOp2B
dense_328/MatMul/ReadVariableOpdense_328/MatMul/ReadVariableOp2D
 dense_329/BiasAdd/ReadVariableOp dense_329/BiasAdd/ReadVariableOp2B
dense_329/MatMul/ReadVariableOpdense_329/MatMul/ReadVariableOp2T
(gru_59/gru_cell_59/MatMul/ReadVariableOp(gru_59/gru_cell_59/MatMul/ReadVariableOp2X
*gru_59/gru_cell_59/MatMul_1/ReadVariableOp*gru_59/gru_cell_59/MatMul_1/ReadVariableOp2F
!gru_59/gru_cell_59/ReadVariableOp!gru_59/gru_cell_59/ReadVariableOp2
gru_59/whilegru_59/while2Z
+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp+lstm_68/lstm_cell_68/BiasAdd/ReadVariableOp2X
*lstm_68/lstm_cell_68/MatMul/ReadVariableOp*lstm_68/lstm_cell_68/MatMul/ReadVariableOp2\
,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp,lstm_68/lstm_cell_68/MatMul_1/ReadVariableOp2
lstm_68/whilelstm_68/while2Z
+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp+lstm_69/lstm_cell_69/BiasAdd/ReadVariableOp2X
*lstm_69/lstm_cell_69/MatMul/ReadVariableOp*lstm_69/lstm_cell_69/MatMul/ReadVariableOp2\
,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp,lstm_69/lstm_cell_69/MatMul_1/ReadVariableOp2
lstm_69/whilelstm_69/while:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б[
і
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188863
inputs_0/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50188778*
condR
while_cond_50188777*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Т
Я
/__inference_lstm_cell_68_layer_call_fn_50190758

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_501843792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
Ь
Ў
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50184988

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
Щ[
є
E__inference_lstm_68_layer_call_and_return_conditional_losses_50186244

inputs/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50186159*
condR
while_cond_50186158*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њG
А
while_body_50189468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_59_readvariableop_resource_06
2while_gru_cell_59_matmul_readvariableop_resource_08
4while_gru_cell_59_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_59_readvariableop_resource4
0while_gru_cell_59_matmul_readvariableop_resource6
2while_gru_cell_59_matmul_1_readvariableop_resourceЂ'while/gru_cell_59/MatMul/ReadVariableOpЂ)while/gru_cell_59/MatMul_1/ReadVariableOpЂ while/gru_cell_59/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
 while/gru_cell_59/ReadVariableOpReadVariableOp+while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_59/ReadVariableOpЂ
while/gru_cell_59/unstackUnpack(while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_59/unstackЦ
'while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'while/gru_cell_59/MatMul/ReadVariableOpд
while/gru_cell_59/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMulМ
while/gru_cell_59/BiasAddBiasAdd"while/gru_cell_59/MatMul:product:0"while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAddt
while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_59/Const
!while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_59/split/split_dimє
while/gru_cell_59/splitSplit*while/gru_cell_59/split/split_dim:output:0"while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/splitЬ
)while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02+
)while/gru_cell_59/MatMul_1/ReadVariableOpН
while/gru_cell_59/MatMul_1MatMulwhile_placeholder_21while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/MatMul_1Т
while/gru_cell_59/BiasAdd_1BiasAdd$while/gru_cell_59/MatMul_1:product:0"while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_59/BiasAdd_1
while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_59/Const_1
#while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_59/split_1/split_dim­
while/gru_cell_59/split_1SplitV$while/gru_cell_59/BiasAdd_1:output:0"while/gru_cell_59/Const_1:output:0,while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_59/split_1Џ
while/gru_cell_59/addAddV2 while/gru_cell_59/split:output:0"while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add
while/gru_cell_59/SigmoidSigmoidwhile/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/SigmoidГ
while/gru_cell_59/add_1AddV2 while/gru_cell_59/split:output:1"while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_1
while/gru_cell_59/Sigmoid_1Sigmoidwhile/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Sigmoid_1Ќ
while/gru_cell_59/mulMulwhile/gru_cell_59/Sigmoid_1:y:0"while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mulЊ
while/gru_cell_59/add_2AddV2 while/gru_cell_59/split:output:2while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_2
while/gru_cell_59/ReluReluwhile/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/Relu
while/gru_cell_59/mul_1Mulwhile/gru_cell_59/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_1w
while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_59/sub/xЈ
while/gru_cell_59/subSub while/gru_cell_59/sub/x:output:0while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/subЌ
while/gru_cell_59/mul_2Mulwhile/gru_cell_59/sub:z:0$while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/mul_2Ї
while/gru_cell_59/add_3AddV2while/gru_cell_59/mul_1:z:0while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_59/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1з
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityъ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1й
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/gru_cell_59/add_3:z:0(^while/gru_cell_59/MatMul/ReadVariableOp*^while/gru_cell_59/MatMul_1/ReadVariableOp!^while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"j
2while_gru_cell_59_matmul_1_readvariableop_resource4while_gru_cell_59_matmul_1_readvariableop_resource_0"f
0while_gru_cell_59_matmul_readvariableop_resource2while_gru_cell_59_matmul_readvariableop_resource_0"X
)while_gru_cell_59_readvariableop_resource+while_gru_cell_59_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2R
'while/gru_cell_59/MatMul/ReadVariableOp'while/gru_cell_59/MatMul/ReadVariableOp2V
)while/gru_cell_59/MatMul_1/ReadVariableOp)while/gru_cell_59/MatMul_1/ReadVariableOp2D
 while/gru_cell_59/ReadVariableOp while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
І

э
lstm_69_while_cond_50187898,
(lstm_69_while_lstm_69_while_loop_counter2
.lstm_69_while_lstm_69_while_maximum_iterations
lstm_69_while_placeholder
lstm_69_while_placeholder_1
lstm_69_while_placeholder_2
lstm_69_while_placeholder_3.
*lstm_69_while_less_lstm_69_strided_slice_1F
Blstm_69_while_lstm_69_while_cond_50187898___redundant_placeholder0F
Blstm_69_while_lstm_69_while_cond_50187898___redundant_placeholder1F
Blstm_69_while_lstm_69_while_cond_50187898___redundant_placeholder2F
Blstm_69_while_lstm_69_while_cond_50187898___redundant_placeholder3
lstm_69_while_identity

lstm_69/while/LessLesslstm_69_while_placeholder*lstm_69_while_less_lstm_69_strided_slice_1*
T0*
_output_shapes
: 2
lstm_69/while/Lessu
lstm_69/while/IdentityIdentitylstm_69/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_69/while/Identity"9
lstm_69_while_identitylstm_69/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к
Д
while_cond_50185286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50185286___redundant_placeholder06
2while_while_cond_50185286___redundant_placeholder16
2while_while_cond_50185286___redundant_placeholder26
2while_while_cond_50185286___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
­
н
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50184412

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates
"
щ
while_body_50185287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0 
while_gru_cell_59_50185309_0 
while_gru_cell_59_50185311_0 
while_gru_cell_59_50185313_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_59_50185309
while_gru_cell_59_50185311
while_gru_cell_59_50185313Ђ)while/gru_cell_59/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemМ
)while/gru_cell_59/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_59_50185309_0while_gru_cell_59_50185311_0while_gru_cell_59_50185313_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501849882+
)while/gru_cell_59/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_59/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_59/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_59/StatefulPartitionedCall:output:1*^while/gru_cell_59/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4":
while_gru_cell_59_50185309while_gru_cell_59_50185309_0":
while_gru_cell_59_50185311while_gru_cell_59_50185311_0":
while_gru_cell_59_50185313while_gru_cell_59_50185313_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2V
)while/gru_cell_59/StatefulPartitionedCall)while/gru_cell_59/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
C

while_body_50187054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
б[
і
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188710
inputs_0/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50188625*
condR
while_cond_50188624*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
C

while_body_50189988
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_50190140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50190140___redundant_placeholder06
2while_while_cond_50190140___redundant_placeholder16
2while_while_cond_50190140___redundant_placeholder26
2while_while_cond_50190140___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
S
	
gru_59_while_body_50188224*
&gru_59_while_gru_59_while_loop_counter0
,gru_59_while_gru_59_while_maximum_iterations
gru_59_while_placeholder
gru_59_while_placeholder_1
gru_59_while_placeholder_2)
%gru_59_while_gru_59_strided_slice_1_0e
agru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_06
2gru_59_while_gru_cell_59_readvariableop_resource_0=
9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0?
;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0
gru_59_while_identity
gru_59_while_identity_1
gru_59_while_identity_2
gru_59_while_identity_3
gru_59_while_identity_4'
#gru_59_while_gru_59_strided_slice_1c
_gru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor4
0gru_59_while_gru_cell_59_readvariableop_resource;
7gru_59_while_gru_cell_59_matmul_readvariableop_resource=
9gru_59_while_gru_cell_59_matmul_1_readvariableop_resourceЂ.gru_59/while/gru_cell_59/MatMul/ReadVariableOpЂ0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpЂ'gru_59/while/gru_cell_59/ReadVariableOpб
>gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2@
>gru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0gru_59/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_0gru_59_while_placeholderGgru_59/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype022
0gru_59/while/TensorArrayV2Read/TensorListGetItemЦ
'gru_59/while/gru_cell_59/ReadVariableOpReadVariableOp2gru_59_while_gru_cell_59_readvariableop_resource_0*
_output_shapes
:	*
dtype02)
'gru_59/while/gru_cell_59/ReadVariableOpЗ
 gru_59/while/gru_cell_59/unstackUnpack/gru_59/while/gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2"
 gru_59/while/gru_cell_59/unstackл
.gru_59/while/gru_cell_59/MatMul/ReadVariableOpReadVariableOp9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.gru_59/while/gru_cell_59/MatMul/ReadVariableOp№
gru_59/while/gru_cell_59/MatMulMatMul7gru_59/while/TensorArrayV2Read/TensorListGetItem:item:06gru_59/while/gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
gru_59/while/gru_cell_59/MatMulи
 gru_59/while/gru_cell_59/BiasAddBiasAdd)gru_59/while/gru_cell_59/MatMul:product:0)gru_59/while/gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 gru_59/while/gru_cell_59/BiasAdd
gru_59/while/gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_59/while/gru_cell_59/Const
(gru_59/while/gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(gru_59/while/gru_cell_59/split/split_dim
gru_59/while/gru_cell_59/splitSplit1gru_59/while/gru_cell_59/split/split_dim:output:0)gru_59/while/gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2 
gru_59/while/gru_cell_59/splitс
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype022
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOpй
!gru_59/while/gru_cell_59/MatMul_1MatMulgru_59_while_placeholder_28gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2#
!gru_59/while/gru_cell_59/MatMul_1о
"gru_59/while/gru_cell_59/BiasAdd_1BiasAdd+gru_59/while/gru_cell_59/MatMul_1:product:0)gru_59/while/gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2$
"gru_59/while/gru_cell_59/BiasAdd_1
 gru_59/while/gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2"
 gru_59/while/gru_cell_59/Const_1Ѓ
*gru_59/while/gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*gru_59/while/gru_cell_59/split_1/split_dimа
 gru_59/while/gru_cell_59/split_1SplitV+gru_59/while/gru_cell_59/BiasAdd_1:output:0)gru_59/while/gru_cell_59/Const_1:output:03gru_59/while/gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 gru_59/while/gru_cell_59/split_1Ы
gru_59/while/gru_cell_59/addAddV2'gru_59/while/gru_cell_59/split:output:0)gru_59/while/gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/addЃ
 gru_59/while/gru_cell_59/SigmoidSigmoid gru_59/while/gru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 gru_59/while/gru_cell_59/SigmoidЯ
gru_59/while/gru_cell_59/add_1AddV2'gru_59/while/gru_cell_59/split:output:1)gru_59/while/gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_1Љ
"gru_59/while/gru_cell_59/Sigmoid_1Sigmoid"gru_59/while/gru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"gru_59/while/gru_cell_59/Sigmoid_1Ш
gru_59/while/gru_cell_59/mulMul&gru_59/while/gru_cell_59/Sigmoid_1:y:0)gru_59/while/gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/mulЦ
gru_59/while/gru_cell_59/add_2AddV2'gru_59/while/gru_cell_59/split:output:2 gru_59/while/gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_2
gru_59/while/gru_cell_59/ReluRelu"gru_59/while/gru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/ReluЛ
gru_59/while/gru_cell_59/mul_1Mul$gru_59/while/gru_cell_59/Sigmoid:y:0gru_59_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/mul_1
gru_59/while/gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
gru_59/while/gru_cell_59/sub/xФ
gru_59/while/gru_cell_59/subSub'gru_59/while/gru_cell_59/sub/x:output:0$gru_59/while/gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/gru_cell_59/subШ
gru_59/while/gru_cell_59/mul_2Mul gru_59/while/gru_cell_59/sub:z:0+gru_59/while/gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/mul_2У
gru_59/while/gru_cell_59/add_3AddV2"gru_59/while/gru_cell_59/mul_1:z:0"gru_59/while/gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_59/while/gru_cell_59/add_3
1gru_59/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_59_while_placeholder_1gru_59_while_placeholder"gru_59/while/gru_cell_59/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_59/while/TensorArrayV2Write/TensorListSetItemj
gru_59/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/while/add/y
gru_59/while/addAddV2gru_59_while_placeholdergru_59/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_59/while/addn
gru_59/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_59/while/add_1/y
gru_59/while/add_1AddV2&gru_59_while_gru_59_while_loop_countergru_59/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_59/while/add_1
gru_59/while/IdentityIdentitygru_59/while/add_1:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity
gru_59/while/Identity_1Identity,gru_59_while_gru_59_while_maximum_iterations/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_1
gru_59/while/Identity_2Identitygru_59/while/add:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_2А
gru_59/while/Identity_3IdentityAgru_59/while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*
_output_shapes
: 2
gru_59/while/Identity_3Ђ
gru_59/while/Identity_4Identity"gru_59/while/gru_cell_59/add_3:z:0/^gru_59/while/gru_cell_59/MatMul/ReadVariableOp1^gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp(^gru_59/while/gru_cell_59/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_59/while/Identity_4"L
#gru_59_while_gru_59_strided_slice_1%gru_59_while_gru_59_strided_slice_1_0"x
9gru_59_while_gru_cell_59_matmul_1_readvariableop_resource;gru_59_while_gru_cell_59_matmul_1_readvariableop_resource_0"t
7gru_59_while_gru_cell_59_matmul_readvariableop_resource9gru_59_while_gru_cell_59_matmul_readvariableop_resource_0"f
0gru_59_while_gru_cell_59_readvariableop_resource2gru_59_while_gru_cell_59_readvariableop_resource_0"7
gru_59_while_identitygru_59/while/Identity:output:0";
gru_59_while_identity_1 gru_59/while/Identity_1:output:0";
gru_59_while_identity_2 gru_59/while/Identity_2:output:0";
gru_59_while_identity_3 gru_59/while/Identity_3:output:0";
gru_59_while_identity_4 gru_59/while/Identity_4:output:0"Ф
_gru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensoragru_59_while_tensorarrayv2read_tensorlistgetitem_gru_59_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2`
.gru_59/while/gru_cell_59/MatMul/ReadVariableOp.gru_59/while/gru_cell_59/MatMul/ReadVariableOp2d
0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp0gru_59/while/gru_cell_59/MatMul_1/ReadVariableOp2R
'gru_59/while/gru_cell_59/ReadVariableOp'gru_59/while/gru_cell_59/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Њ

Ш
*__inference_model_9_layer_call_fn_50187388
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_501873552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Е
Э
while_cond_50188952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50188952___redundant_placeholder06
2while_while_cond_50188952___redundant_placeholder16
2while_while_cond_50188952___redundant_placeholder26
2while_while_cond_50188952___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
C

while_body_50190141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_69_matmul_readvariableop_resource_09
5while_lstm_cell_69_matmul_1_readvariableop_resource_08
4while_lstm_cell_69_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_69_matmul_readvariableop_resource7
3while_lstm_cell_69_matmul_1_readvariableop_resource6
2while_lstm_cell_69_biasadd_readvariableop_resourceЂ)while/lstm_cell_69/BiasAdd/ReadVariableOpЂ(while/lstm_cell_69/MatMul/ReadVariableOpЂ*while/lstm_cell_69/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_69/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_69_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_69/MatMul/ReadVariableOpз
while/lstm_cell_69/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMulЯ
*while/lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_69_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_69/MatMul_1/ReadVariableOpР
while/lstm_cell_69/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/MatMul_1И
while/lstm_cell_69/addAddV2#while/lstm_cell_69/MatMul:product:0%while/lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/addШ
)while/lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_69_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_69/BiasAdd/ReadVariableOpХ
while/lstm_cell_69/BiasAddBiasAddwhile/lstm_cell_69/add:z:01while/lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_69/BiasAddv
while/lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_69/Const
"while/lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_69/split/split_dim
while/lstm_cell_69/splitSplit+while/lstm_cell_69/split/split_dim:output:0#while/lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_69/split
while/lstm_cell_69/SigmoidSigmoid!while/lstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid
while/lstm_cell_69/Sigmoid_1Sigmoid!while/lstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_1 
while/lstm_cell_69/mulMul while/lstm_cell_69/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul
while/lstm_cell_69/ReluRelu!while/lstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/ReluД
while/lstm_cell_69/mul_1Mulwhile/lstm_cell_69/Sigmoid:y:0%while/lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_1Љ
while/lstm_cell_69/add_1AddV2while/lstm_cell_69/mul:z:0while/lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/add_1
while/lstm_cell_69/Sigmoid_2Sigmoid!while/lstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Sigmoid_2
while/lstm_cell_69/Relu_1Reluwhile/lstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/Relu_1И
while/lstm_cell_69/mul_2Mul while/lstm_cell_69/Sigmoid_2:y:0'while/lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_69/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_69/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_69/mul_2:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_69/add_1:z:0*^while/lstm_cell_69/BiasAdd/ReadVariableOp)^while/lstm_cell_69/MatMul/ReadVariableOp+^while/lstm_cell_69/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_69_biasadd_readvariableop_resource4while_lstm_cell_69_biasadd_readvariableop_resource_0"l
3while_lstm_cell_69_matmul_1_readvariableop_resource5while_lstm_cell_69_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_69_matmul_readvariableop_resource3while_lstm_cell_69_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_69/BiasAdd/ReadVariableOp)while/lstm_cell_69/BiasAdd/ReadVariableOp2T
(while/lstm_cell_69/MatMul/ReadVariableOp(while/lstm_cell_69/MatMul/ReadVariableOp2X
*while/lstm_cell_69/MatMul_1/ReadVariableOp*while/lstm_cell_69/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Р%

while_body_50185878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_69_50185902_0!
while_lstm_cell_69_50185904_0!
while_lstm_cell_69_50185906_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_69_50185902
while_lstm_cell_69_50185904
while_lstm_cell_69_50185906Ђ*while/lstm_cell_69/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_69/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_69_50185902_0while_lstm_cell_69_50185904_0while_lstm_cell_69_50185906_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855512,
*while/lstm_cell_69/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_69/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_69/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_69/StatefulPartitionedCall:output:1+^while/lstm_cell_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_69/StatefulPartitionedCall:output:2+^while/lstm_cell_69/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_69_50185902while_lstm_cell_69_50185902_0"<
while_lstm_cell_69_50185904while_lstm_cell_69_50185904_0"<
while_lstm_cell_69_50185906while_lstm_cell_69_50185906_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2X
*while/lstm_cell_69/StatefulPartitionedCall*while/lstm_cell_69/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
І

э
lstm_68_while_cond_50188073,
(lstm_68_while_lstm_68_while_loop_counter2
.lstm_68_while_lstm_68_while_maximum_iterations
lstm_68_while_placeholder
lstm_68_while_placeholder_1
lstm_68_while_placeholder_2
lstm_68_while_placeholder_3.
*lstm_68_while_less_lstm_68_strided_slice_1F
Blstm_68_while_lstm_68_while_cond_50188073___redundant_placeholder0F
Blstm_68_while_lstm_68_while_cond_50188073___redundant_placeholder1F
Blstm_68_while_lstm_68_while_cond_50188073___redundant_placeholder2F
Blstm_68_while_lstm_68_while_cond_50188073___redundant_placeholder3
lstm_68_while_identity

lstm_68/while/LessLesslstm_68_while_placeholder*lstm_68_while_less_lstm_68_strided_slice_1*
T0*
_output_shapes
: 2
lstm_68/while/Lessu
lstm_68/while/IdentityIdentitylstm_68/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_68/while/Identity"9
lstm_68_while_identitylstm_68/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
ц

,__inference_dense_329_layer_call_fn_50190675

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_329_layer_call_and_return_conditional_losses_501872492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Е
Э
while_cond_50186311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50186311___redundant_placeholder06
2while_while_cond_50186311___redundant_placeholder16
2while_while_cond_50186311___redundant_placeholder26
2while_while_cond_50186311___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
Е
п
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190949

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Ђ

*__inference_lstm_69_layer_call_fn_50190248
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_69_layer_call_and_return_conditional_losses_501860792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
 

)__inference_gru_59_layer_call_fn_50189909
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501853512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
­
н
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50185584

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
я
g
H__inference_dropout_68_layer_call_and_return_conditional_losses_50186786

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeС
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Е
Э
while_cond_50189105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50189105___redundant_placeholder06
2while_while_cond_50189105___redundant_placeholder16
2while_while_cond_50189105___redundant_placeholder26
2while_while_cond_50189105___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
јD
ф
E__inference_lstm_69_layer_call_and_return_conditional_losses_50185947

inputs
lstm_cell_69_50185865
lstm_cell_69_50185867
lstm_cell_69_50185869
identityЂ$lstm_cell_69/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_69/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_69_50185865lstm_cell_69_50185867lstm_cell_69_50185869*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_501855512&
$lstm_cell_69/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_69_50185865lstm_cell_69_50185867lstm_cell_69_50185869*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50185878*
condR
while_cond_50185877*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_69/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2L
$lstm_cell_69/StatefulPartitionedCall$lstm_cell_69/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
в[
н
D__inference_gru_59_layer_call_and_return_conditional_losses_50189558

inputs'
#gru_cell_59_readvariableop_resource.
*gru_cell_59_matmul_readvariableop_resource0
,gru_cell_59_matmul_1_readvariableop_resource
identityЂ!gru_cell_59/MatMul/ReadVariableOpЂ#gru_cell_59/MatMul_1/ReadVariableOpЂgru_cell_59/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_59/ReadVariableOpReadVariableOp#gru_cell_59_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_59/ReadVariableOp
gru_cell_59/unstackUnpack"gru_cell_59/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_59/unstackВ
!gru_cell_59/MatMul/ReadVariableOpReadVariableOp*gru_cell_59_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!gru_cell_59/MatMul/ReadVariableOpЊ
gru_cell_59/MatMulMatMulstrided_slice_2:output:0)gru_cell_59/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMulЄ
gru_cell_59/BiasAddBiasAddgru_cell_59/MatMul:product:0gru_cell_59/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAddh
gru_cell_59/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_59/Const
gru_cell_59/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split/split_dimм
gru_cell_59/splitSplit$gru_cell_59/split/split_dim:output:0gru_cell_59/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/splitИ
#gru_cell_59/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_59_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02%
#gru_cell_59/MatMul_1/ReadVariableOpІ
gru_cell_59/MatMul_1MatMulzeros:output:0+gru_cell_59/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/MatMul_1Њ
gru_cell_59/BiasAdd_1BiasAddgru_cell_59/MatMul_1:product:0gru_cell_59/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_59/BiasAdd_1
gru_cell_59/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_59/Const_1
gru_cell_59/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_59/split_1/split_dim
gru_cell_59/split_1SplitVgru_cell_59/BiasAdd_1:output:0gru_cell_59/Const_1:output:0&gru_cell_59/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_59/split_1
gru_cell_59/addAddV2gru_cell_59/split:output:0gru_cell_59/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add|
gru_cell_59/SigmoidSigmoidgru_cell_59/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid
gru_cell_59/add_1AddV2gru_cell_59/split:output:1gru_cell_59/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_1
gru_cell_59/Sigmoid_1Sigmoidgru_cell_59/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Sigmoid_1
gru_cell_59/mulMulgru_cell_59/Sigmoid_1:y:0gru_cell_59/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul
gru_cell_59/add_2AddV2gru_cell_59/split:output:2gru_cell_59/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_2u
gru_cell_59/ReluRelugru_cell_59/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/Relu
gru_cell_59/mul_1Mulgru_cell_59/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_1k
gru_cell_59/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_59/sub/x
gru_cell_59/subSubgru_cell_59/sub/x:output:0gru_cell_59/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/sub
gru_cell_59/mul_2Mulgru_cell_59/sub:z:0gru_cell_59/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/mul_2
gru_cell_59/add_3AddV2gru_cell_59/mul_1:z:0gru_cell_59/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_59/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_59_readvariableop_resource*gru_cell_59_matmul_readvariableop_resource,gru_cell_59_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50189468*
condR
while_cond_50189467*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeл
IdentityIdentitystrided_slice_3:output:0"^gru_cell_59/MatMul/ReadVariableOp$^gru_cell_59/MatMul_1/ReadVariableOp^gru_cell_59/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!gru_cell_59/MatMul/ReadVariableOp!gru_cell_59/MatMul/ReadVariableOp2J
#gru_cell_59/MatMul_1/ReadVariableOp#gru_cell_59/MatMul_1/ReadVariableOp28
gru_cell_59/ReadVariableOpgru_cell_59/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
Э
while_cond_50186158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50186158___redundant_placeholder06
2while_while_cond_50186158___redundant_placeholder16
2while_while_cond_50186158___redundant_placeholder26
2while_while_cond_50186158___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:


#model_9_lstm_68_while_cond_50183892<
8model_9_lstm_68_while_model_9_lstm_68_while_loop_counterB
>model_9_lstm_68_while_model_9_lstm_68_while_maximum_iterations%
!model_9_lstm_68_while_placeholder'
#model_9_lstm_68_while_placeholder_1'
#model_9_lstm_68_while_placeholder_2'
#model_9_lstm_68_while_placeholder_3>
:model_9_lstm_68_while_less_model_9_lstm_68_strided_slice_1V
Rmodel_9_lstm_68_while_model_9_lstm_68_while_cond_50183892___redundant_placeholder0V
Rmodel_9_lstm_68_while_model_9_lstm_68_while_cond_50183892___redundant_placeholder1V
Rmodel_9_lstm_68_while_model_9_lstm_68_while_cond_50183892___redundant_placeholder2V
Rmodel_9_lstm_68_while_model_9_lstm_68_while_cond_50183892___redundant_placeholder3"
model_9_lstm_68_while_identity
Р
model_9/lstm_68/while/LessLess!model_9_lstm_68_while_placeholder:model_9_lstm_68_while_less_model_9_lstm_68_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_68/while/Less
model_9/lstm_68/while/IdentityIdentitymodel_9/lstm_68/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_9/lstm_68/while/Identity"I
model_9_lstm_68_while_identity'model_9/lstm_68/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
я
g
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189225

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeС
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs


)__inference_gru_59_layer_call_fn_50189569

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_59_layer_call_and_return_conditional_losses_501865852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ[
є
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189191

inputs/
+lstm_cell_68_matmul_readvariableop_resource1
-lstm_cell_68_matmul_1_readvariableop_resource0
,lstm_cell_68_biasadd_readvariableop_resource
identityЂ#lstm_cell_68/BiasAdd/ReadVariableOpЂ"lstm_cell_68/MatMul/ReadVariableOpЂ$lstm_cell_68/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_68/MatMul/ReadVariableOpReadVariableOp+lstm_cell_68_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_68/MatMul/ReadVariableOp­
lstm_cell_68/MatMulMatMulstrided_slice_2:output:0*lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMulЛ
$lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_68_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_68/MatMul_1/ReadVariableOpЉ
lstm_cell_68/MatMul_1MatMulzeros:output:0,lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/MatMul_1 
lstm_cell_68/addAddV2lstm_cell_68/MatMul:product:0lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/addД
#lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_68_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_68/BiasAdd/ReadVariableOp­
lstm_cell_68/BiasAddBiasAddlstm_cell_68/add:z:0+lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_68/BiasAddj
lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/Const~
lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_68/split/split_dimѓ
lstm_cell_68/splitSplit%lstm_cell_68/split/split_dim:output:0lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_68/split
lstm_cell_68/SigmoidSigmoidlstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid
lstm_cell_68/Sigmoid_1Sigmoidlstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_1
lstm_cell_68/mulMullstm_cell_68/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul}
lstm_cell_68/ReluRelulstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu
lstm_cell_68/mul_1Mullstm_cell_68/Sigmoid:y:0lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_1
lstm_cell_68/add_1AddV2lstm_cell_68/mul:z:0lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/add_1
lstm_cell_68/Sigmoid_2Sigmoidlstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Sigmoid_2|
lstm_cell_68/Relu_1Relulstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/Relu_1 
lstm_cell_68/mul_2Mullstm_cell_68/Sigmoid_2:y:0!lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_68/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_68_matmul_readvariableop_resource-lstm_cell_68_matmul_1_readvariableop_resource,lstm_cell_68_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50189106*
condR
while_cond_50189105*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_68/BiasAdd/ReadVariableOp#^lstm_cell_68/MatMul/ReadVariableOp%^lstm_cell_68/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_68/BiasAdd/ReadVariableOp#lstm_cell_68/BiasAdd/ReadVariableOp2H
"lstm_cell_68/MatMul/ReadVariableOp"lstm_cell_68/MatMul/ReadVariableOp2L
$lstm_cell_68/MatMul_1/ReadVariableOp$lstm_cell_68/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э[
і
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190073
inputs_0/
+lstm_cell_69_matmul_readvariableop_resource1
-lstm_cell_69_matmul_1_readvariableop_resource0
,lstm_cell_69_biasadd_readvariableop_resource
identityЂ#lstm_cell_69/BiasAdd/ReadVariableOpЂ"lstm_cell_69/MatMul/ReadVariableOpЂ$lstm_cell_69/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_69/MatMul/ReadVariableOpReadVariableOp+lstm_cell_69_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_69/MatMul/ReadVariableOp­
lstm_cell_69/MatMulMatMulstrided_slice_2:output:0*lstm_cell_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMulЛ
$lstm_cell_69/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_69_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_69/MatMul_1/ReadVariableOpЉ
lstm_cell_69/MatMul_1MatMulzeros:output:0,lstm_cell_69/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/MatMul_1 
lstm_cell_69/addAddV2lstm_cell_69/MatMul:product:0lstm_cell_69/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/addД
#lstm_cell_69/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_69_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_69/BiasAdd/ReadVariableOp­
lstm_cell_69/BiasAddBiasAddlstm_cell_69/add:z:0+lstm_cell_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_69/BiasAddj
lstm_cell_69/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/Const~
lstm_cell_69/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_69/split/split_dimѓ
lstm_cell_69/splitSplit%lstm_cell_69/split/split_dim:output:0lstm_cell_69/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_69/split
lstm_cell_69/SigmoidSigmoidlstm_cell_69/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid
lstm_cell_69/Sigmoid_1Sigmoidlstm_cell_69/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_1
lstm_cell_69/mulMullstm_cell_69/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul}
lstm_cell_69/ReluRelulstm_cell_69/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu
lstm_cell_69/mul_1Mullstm_cell_69/Sigmoid:y:0lstm_cell_69/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_1
lstm_cell_69/add_1AddV2lstm_cell_69/mul:z:0lstm_cell_69/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/add_1
lstm_cell_69/Sigmoid_2Sigmoidlstm_cell_69/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Sigmoid_2|
lstm_cell_69/Relu_1Relulstm_cell_69/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/Relu_1 
lstm_cell_69/mul_2Mullstm_cell_69/Sigmoid_2:y:0!lstm_cell_69/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_69/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_69_matmul_readvariableop_resource-lstm_cell_69_matmul_1_readvariableop_resource,lstm_cell_69_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_50189988*
condR
while_cond_50189987*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_69/BiasAdd/ReadVariableOp#^lstm_cell_69/MatMul/ReadVariableOp%^lstm_cell_69/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_69/BiasAdd/ReadVariableOp#lstm_cell_69/BiasAdd/ReadVariableOp2H
"lstm_cell_69/MatMul/ReadVariableOp"lstm_cell_69/MatMul/ReadVariableOp2L
$lstm_cell_69/MatMul_1/ReadVariableOp$lstm_cell_69/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
ё	
р
G__inference_dense_327_layer_call_and_return_conditional_losses_50187180

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Е
Э
while_cond_50189987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_50189987___redundant_placeholder06
2while_while_cond_50189987___redundant_placeholder16
2while_while_cond_50189987___redundant_placeholder26
2while_while_cond_50189987___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
І	
Й
gru_59_while_cond_50188223*
&gru_59_while_gru_59_while_loop_counter0
,gru_59_while_gru_59_while_maximum_iterations
gru_59_while_placeholder
gru_59_while_placeholder_1
gru_59_while_placeholder_2,
(gru_59_while_less_gru_59_strided_slice_1D
@gru_59_while_gru_59_while_cond_50188223___redundant_placeholder0D
@gru_59_while_gru_59_while_cond_50188223___redundant_placeholder1D
@gru_59_while_gru_59_while_cond_50188223___redundant_placeholder2D
@gru_59_while_gru_59_while_cond_50188223___redundant_placeholder3
gru_59_while_identity

gru_59/while/LessLessgru_59_while_placeholder(gru_59_while_less_gru_59_strided_slice_1*
T0*
_output_shapes
: 2
gru_59/while/Lessr
gru_59/while/IdentityIdentitygru_59/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_59/while/Identity"7
gru_59_while_identitygru_59/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
І	
Й
gru_59_while_cond_50187728*
&gru_59_while_gru_59_while_loop_counter0
,gru_59_while_gru_59_while_maximum_iterations
gru_59_while_placeholder
gru_59_while_placeholder_1
gru_59_while_placeholder_2,
(gru_59_while_less_gru_59_strided_slice_1D
@gru_59_while_gru_59_while_cond_50187728___redundant_placeholder0D
@gru_59_while_gru_59_while_cond_50187728___redundant_placeholder1D
@gru_59_while_gru_59_while_cond_50187728___redundant_placeholder2D
@gru_59_while_gru_59_while_cond_50187728___redundant_placeholder3
gru_59_while_identity

gru_59/while/LessLessgru_59_while_placeholder(gru_59_while_less_gru_59_strided_slice_1*
T0*
_output_shapes
: 2
gru_59/while/Lessr
gru_59/while/IdentityIdentitygru_59/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_59/while/Identity"7
gru_59_while_identitygru_59/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
кt
і
!__inference__traced_save_50191174
file_prefix/
+savev2_dense_327_kernel_read_readvariableop-
)savev2_dense_327_bias_read_readvariableop/
+savev2_dense_328_kernel_read_readvariableop-
)savev2_dense_328_bias_read_readvariableop/
+savev2_dense_329_kernel_read_readvariableop-
)savev2_dense_329_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_68_lstm_cell_68_kernel_read_readvariableopD
@savev2_lstm_68_lstm_cell_68_recurrent_kernel_read_readvariableop8
4savev2_lstm_68_lstm_cell_68_bias_read_readvariableop8
4savev2_gru_59_gru_cell_59_kernel_read_readvariableopB
>savev2_gru_59_gru_cell_59_recurrent_kernel_read_readvariableop6
2savev2_gru_59_gru_cell_59_bias_read_readvariableop:
6savev2_lstm_69_lstm_cell_69_kernel_read_readvariableopD
@savev2_lstm_69_lstm_cell_69_recurrent_kernel_read_readvariableop8
4savev2_lstm_69_lstm_cell_69_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_dense_327_kernel_m_read_readvariableop4
0savev2_adam_dense_327_bias_m_read_readvariableop6
2savev2_adam_dense_328_kernel_m_read_readvariableop4
0savev2_adam_dense_328_bias_m_read_readvariableop6
2savev2_adam_dense_329_kernel_m_read_readvariableop4
0savev2_adam_dense_329_bias_m_read_readvariableopA
=savev2_adam_lstm_68_lstm_cell_68_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_68_lstm_cell_68_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_68_lstm_cell_68_bias_m_read_readvariableop?
;savev2_adam_gru_59_gru_cell_59_kernel_m_read_readvariableopI
Esavev2_adam_gru_59_gru_cell_59_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_59_gru_cell_59_bias_m_read_readvariableopA
=savev2_adam_lstm_69_lstm_cell_69_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_69_lstm_cell_69_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_69_lstm_cell_69_bias_m_read_readvariableop6
2savev2_adam_dense_327_kernel_v_read_readvariableop4
0savev2_adam_dense_327_bias_v_read_readvariableop6
2savev2_adam_dense_328_kernel_v_read_readvariableop4
0savev2_adam_dense_328_bias_v_read_readvariableop6
2savev2_adam_dense_329_kernel_v_read_readvariableop4
0savev2_adam_dense_329_bias_v_read_readvariableopA
=savev2_adam_lstm_68_lstm_cell_68_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_68_lstm_cell_68_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_68_lstm_cell_68_bias_v_read_readvariableop?
;savev2_adam_gru_59_gru_cell_59_kernel_v_read_readvariableopI
Esavev2_adam_gru_59_gru_cell_59_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_59_gru_cell_59_bias_v_read_readvariableopA
=savev2_adam_lstm_69_lstm_cell_69_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_69_lstm_cell_69_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_69_lstm_cell_69_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЖ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Ш
valueОBЛ9B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_327_kernel_read_readvariableop)savev2_dense_327_bias_read_readvariableop+savev2_dense_328_kernel_read_readvariableop)savev2_dense_328_bias_read_readvariableop+savev2_dense_329_kernel_read_readvariableop)savev2_dense_329_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_68_lstm_cell_68_kernel_read_readvariableop@savev2_lstm_68_lstm_cell_68_recurrent_kernel_read_readvariableop4savev2_lstm_68_lstm_cell_68_bias_read_readvariableop4savev2_gru_59_gru_cell_59_kernel_read_readvariableop>savev2_gru_59_gru_cell_59_recurrent_kernel_read_readvariableop2savev2_gru_59_gru_cell_59_bias_read_readvariableop6savev2_lstm_69_lstm_cell_69_kernel_read_readvariableop@savev2_lstm_69_lstm_cell_69_recurrent_kernel_read_readvariableop4savev2_lstm_69_lstm_cell_69_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_327_kernel_m_read_readvariableop0savev2_adam_dense_327_bias_m_read_readvariableop2savev2_adam_dense_328_kernel_m_read_readvariableop0savev2_adam_dense_328_bias_m_read_readvariableop2savev2_adam_dense_329_kernel_m_read_readvariableop0savev2_adam_dense_329_bias_m_read_readvariableop=savev2_adam_lstm_68_lstm_cell_68_kernel_m_read_readvariableopGsavev2_adam_lstm_68_lstm_cell_68_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_68_lstm_cell_68_bias_m_read_readvariableop;savev2_adam_gru_59_gru_cell_59_kernel_m_read_readvariableopEsavev2_adam_gru_59_gru_cell_59_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_59_gru_cell_59_bias_m_read_readvariableop=savev2_adam_lstm_69_lstm_cell_69_kernel_m_read_readvariableopGsavev2_adam_lstm_69_lstm_cell_69_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_69_lstm_cell_69_bias_m_read_readvariableop2savev2_adam_dense_327_kernel_v_read_readvariableop0savev2_adam_dense_327_bias_v_read_readvariableop2savev2_adam_dense_328_kernel_v_read_readvariableop0savev2_adam_dense_328_bias_v_read_readvariableop2savev2_adam_dense_329_kernel_v_read_readvariableop0savev2_adam_dense_329_bias_v_read_readvariableop=savev2_adam_lstm_68_lstm_cell_68_kernel_v_read_readvariableopGsavev2_adam_lstm_68_lstm_cell_68_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_68_lstm_cell_68_bias_v_read_readvariableop;savev2_adam_gru_59_gru_cell_59_kernel_v_read_readvariableopEsavev2_adam_gru_59_gru_cell_59_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_59_gru_cell_59_bias_v_read_readvariableop=savev2_adam_lstm_69_lstm_cell_69_kernel_v_read_readvariableopGsavev2_adam_lstm_69_lstm_cell_69_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_69_lstm_cell_69_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *G
dtypes=
;29	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*а
_input_shapesО
Л: :2 : :2@:@:`:: : : : : :	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш: : : : : : :2 : :2@:@:`::	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш:2 : :2@:@:`::	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2 : 

_output_shapes
: :$ 

_output_shapes

:2@: 

_output_shapes
:@:$ 

_output_shapes

:`: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Ќ:%!

_output_shapes
:	KЌ:!

_output_shapes	
:Ќ:%!

_output_shapes
:	:%!

_output_shapes
:	2:%!

_output_shapes
:	:%!

_output_shapes
:	KШ:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2 : 

_output_shapes
: :$ 

_output_shapes

:2@: 

_output_shapes
:@:$ 

_output_shapes

:`:  

_output_shapes
::%!!

_output_shapes
:	Ќ:%"!

_output_shapes
:	KЌ:!#

_output_shapes	
:Ќ:%$!

_output_shapes
:	:%%!

_output_shapes
:	2:%&!

_output_shapes
:	:%'!

_output_shapes
:	KШ:%(!

_output_shapes
:	2Ш:!)

_output_shapes	
:Ш:$* 

_output_shapes

:2 : +

_output_shapes
: :$, 

_output_shapes

:2@: -

_output_shapes
:@:$. 

_output_shapes

:`: /

_output_shapes
::%0!

_output_shapes
:	Ќ:%1!

_output_shapes
:	KЌ:!2

_output_shapes	
:Ќ:%3!

_output_shapes
:	:%4!

_output_shapes
:	2:%5!

_output_shapes
:	:%6!

_output_shapes
:	KШ:%7!

_output_shapes
:	2Ш:!8

_output_shapes	
:Ш:9

_output_shapes
: 
Є

Ц
*__inference_model_9_layer_call_fn_50188522

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_501873552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
=
п
D__inference_gru_59_layer_call_and_return_conditional_losses_50185351

inputs
gru_cell_59_50185275
gru_cell_59_50185277
gru_cell_59_50185279
identityЂ#gru_cell_59/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2ћ
#gru_cell_59/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_59_50185275gru_cell_59_50185277gru_cell_59_50185279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_501849882%
#gru_cell_59/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_59_50185275gru_cell_59_50185277gru_cell_59_50185279*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_50185287*
condR
while_cond_50185286*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^gru_cell_59/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_59/StatefulPartitionedCall#gru_cell_59/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І

э
lstm_69_while_cond_50188379,
(lstm_69_while_lstm_69_while_loop_counter2
.lstm_69_while_lstm_69_while_maximum_iterations
lstm_69_while_placeholder
lstm_69_while_placeholder_1
lstm_69_while_placeholder_2
lstm_69_while_placeholder_3.
*lstm_69_while_less_lstm_69_strided_slice_1F
Blstm_69_while_lstm_69_while_cond_50188379___redundant_placeholder0F
Blstm_69_while_lstm_69_while_cond_50188379___redundant_placeholder1F
Blstm_69_while_lstm_69_while_cond_50188379___redundant_placeholder2F
Blstm_69_while_lstm_69_while_cond_50188379___redundant_placeholder3
lstm_69_while_identity

lstm_69/while/LessLesslstm_69_while_placeholder*lstm_69_while_less_lstm_69_strided_slice_1*
T0*
_output_shapes
: 2
lstm_69/while/Lessu
lstm_69/while/IdentityIdentitylstm_69/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_69/while/Identity"9
lstm_69_while_identitylstm_69/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
д
А
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190855

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
к
Д
while_cond_50189648
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_50189648___redundant_placeholder06
2while_while_cond_50189648___redundant_placeholder16
2while_while_cond_50189648___redundant_placeholder26
2while_while_cond_50189648___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
C

while_body_50188778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_68_matmul_readvariableop_resource_09
5while_lstm_cell_68_matmul_1_readvariableop_resource_08
4while_lstm_cell_68_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_68_matmul_readvariableop_resource7
3while_lstm_cell_68_matmul_1_readvariableop_resource6
2while_lstm_cell_68_biasadd_readvariableop_resourceЂ)while/lstm_cell_68/BiasAdd/ReadVariableOpЂ(while/lstm_cell_68/MatMul/ReadVariableOpЂ*while/lstm_cell_68/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_68/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_68_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_68/MatMul/ReadVariableOpз
while/lstm_cell_68/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMulЯ
*while/lstm_cell_68/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_68_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_68/MatMul_1/ReadVariableOpР
while/lstm_cell_68/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_68/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/MatMul_1И
while/lstm_cell_68/addAddV2#while/lstm_cell_68/MatMul:product:0%while/lstm_cell_68/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/addШ
)while/lstm_cell_68/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_68_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_68/BiasAdd/ReadVariableOpХ
while/lstm_cell_68/BiasAddBiasAddwhile/lstm_cell_68/add:z:01while/lstm_cell_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_68/BiasAddv
while/lstm_cell_68/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_68/Const
"while/lstm_cell_68/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_68/split/split_dim
while/lstm_cell_68/splitSplit+while/lstm_cell_68/split/split_dim:output:0#while/lstm_cell_68/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_68/split
while/lstm_cell_68/SigmoidSigmoid!while/lstm_cell_68/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid
while/lstm_cell_68/Sigmoid_1Sigmoid!while/lstm_cell_68/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_1 
while/lstm_cell_68/mulMul while/lstm_cell_68/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul
while/lstm_cell_68/ReluRelu!while/lstm_cell_68/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/ReluД
while/lstm_cell_68/mul_1Mulwhile/lstm_cell_68/Sigmoid:y:0%while/lstm_cell_68/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_1Љ
while/lstm_cell_68/add_1AddV2while/lstm_cell_68/mul:z:0while/lstm_cell_68/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/add_1
while/lstm_cell_68/Sigmoid_2Sigmoid!while/lstm_cell_68/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Sigmoid_2
while/lstm_cell_68/Relu_1Reluwhile/lstm_cell_68/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/Relu_1И
while/lstm_cell_68/mul_2Mul while/lstm_cell_68/Sigmoid_2:y:0'while/lstm_cell_68/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_68/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_68/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_68/mul_2:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_68/add_1:z:0*^while/lstm_cell_68/BiasAdd/ReadVariableOp)^while/lstm_cell_68/MatMul/ReadVariableOp+^while/lstm_cell_68/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_68_biasadd_readvariableop_resource4while_lstm_cell_68_biasadd_readvariableop_resource_0"l
3while_lstm_cell_68_matmul_1_readvariableop_resource5while_lstm_cell_68_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_68_matmul_readvariableop_resource3while_lstm_cell_68_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_68/BiasAdd/ReadVariableOp)while/lstm_cell_68/BiasAdd/ReadVariableOp2T
(while/lstm_cell_68/MatMul/ReadVariableOp(while/lstm_cell_68/MatMul/ReadVariableOp2X
*while/lstm_cell_68/MatMul_1/ReadVariableOp*while/lstm_cell_68/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: "БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
J
input_10>
serving_default_input_10:0џџџџџџџџџџџџџџџџџџ=
	dense_3290
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ч
л]
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+Ю&call_and_return_all_conditional_losses
Я_default_save_signature
а__call__"яY
_tf_keras_networkгY{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_68", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_68", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_68", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "dropout_68", "inbound_nodes": [[["lstm_68", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_59", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_59", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_69", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_69", "inbound_nodes": [[["dropout_68", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_69", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_69", "inbound_nodes": [[["gru_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_327", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_327", "inbound_nodes": [[["lstm_69", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_328", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_328", "inbound_nodes": [[["dropout_69", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["dense_327", 0, 0, {}], ["dense_328", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_329", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_329", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["dense_329", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_68", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_68", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_68", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "dropout_68", "inbound_nodes": [[["lstm_68", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_59", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_59", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_69", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_69", "inbound_nodes": [[["dropout_68", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_69", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_69", "inbound_nodes": [[["gru_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_327", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_327", "inbound_nodes": [[["lstm_69", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_328", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_328", "inbound_nodes": [[["dropout_69", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["dense_327", 0, 0, {}], ["dense_328", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_329", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_329", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["dense_329", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 9.999999747378752e-05, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ї"є
_tf_keras_input_layerд{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
Р
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+б&call_and_return_all_conditional_losses
в__call__"

_tf_keras_rnn_layerї	{"class_name": "LSTM", "name": "lstm_68", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_68", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
ъ
trainable_variables
	variables
regularization_losses
	keras_api
+г&call_and_return_all_conditional_losses
д__call__"й
_tf_keras_layerП{"class_name": "Dropout", "name": "dropout_68", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_68", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
Й
cell

state_spec
trainable_variables
	variables
regularization_losses
 	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"

_tf_keras_rnn_layer№	{"class_name": "GRU", "name": "gru_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_59", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
У
!cell
"
state_spec
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+з&call_and_return_all_conditional_losses
и__call__"

_tf_keras_rnn_layerњ	{"class_name": "LSTM", "name": "lstm_69", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_69", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 75]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 75]}}
щ
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+й&call_and_return_all_conditional_losses
к__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_69", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_69", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
і

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_327", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_327", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
і

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_328", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_328", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Я
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+п&call_and_return_all_conditional_losses
р__call__"О
_tf_keras_layerЄ{"class_name": "Concatenate", "name": "concatenate_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 64]}]}
ї

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+с&call_and_return_all_conditional_losses
т__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_329", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_329", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96]}}
џ
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate+mА,mБ1mВ2mГ;mД<mЕFmЖGmЗHmИImЙJmКKmЛLmМMmНNmО+vП,vР1vС2vТ;vУ<vФFvХGvЦHvЧIvШJvЩKvЪLvЫMvЬNvЭ"
	optimizer

F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14"
trackable_list_wrapper

F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
trainable_variables
Olayer_metrics
	variables
Pmetrics

Qlayers
Rlayer_regularization_losses
regularization_losses
Snon_trainable_variables
а__call__
Я_default_save_signature
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
-
уserving_default"
signature_map
Ќ

Fkernel
Grecurrent_kernel
Hbias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"я
_tf_keras_layerе{"class_name": "LSTMCell", "name": "lstm_cell_68", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_68", "trainable": true, "dtype": "float32", "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
М
trainable_variables
Xlayer_metrics
	variables

Ystates
Zmetrics

[layers
\layer_regularization_losses
regularization_losses
]non_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
trainable_variables
^layer_metrics
	variables
_metrics

`layers
alayer_regularization_losses
regularization_losses
bnon_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
Є

Ikernel
Jrecurrent_kernel
Kbias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_59", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
М
trainable_variables
glayer_metrics
	variables

hstates
imetrics

jlayers
klayer_regularization_losses
regularization_losses
lnon_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
Ќ

Lkernel
Mrecurrent_kernel
Nbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"я
_tf_keras_layerе{"class_name": "LSTMCell", "name": "lstm_cell_69", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_69", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
М
#trainable_variables
qlayer_metrics
$	variables

rstates
smetrics

tlayers
ulayer_regularization_losses
%regularization_losses
vnon_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
'trainable_variables
wlayer_metrics
(	variables
xmetrics

ylayers
zlayer_regularization_losses
)regularization_losses
{non_trainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
": 2 2dense_327/kernel
: 2dense_327/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
Б
-trainable_variables
|layer_metrics
.	variables
}metrics

~layers
layer_regularization_losses
/regularization_losses
non_trainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
": 2@2dense_328/kernel
:@2dense_328/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
3trainable_variables
layer_metrics
4	variables
metrics
layers
 layer_regularization_losses
5regularization_losses
non_trainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
7trainable_variables
layer_metrics
8	variables
metrics
layers
 layer_regularization_losses
9regularization_losses
non_trainable_variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
": `2dense_329/kernel
:2dense_329/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
=trainable_variables
layer_metrics
>	variables
metrics
layers
 layer_regularization_losses
?regularization_losses
non_trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	Ќ2lstm_68/lstm_cell_68/kernel
8:6	KЌ2%lstm_68/lstm_cell_68/recurrent_kernel
(:&Ќ2lstm_68/lstm_cell_68/bias
,:*	2gru_59/gru_cell_59/kernel
6:4	22#gru_59/gru_cell_59/recurrent_kernel
*:(	2gru_59/gru_cell_59/bias
.:,	KШ2lstm_69/lstm_cell_69/kernel
8:6	2Ш2%lstm_69/lstm_cell_69/recurrent_kernel
(:&Ш2lstm_69/lstm_cell_69/bias
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ttrainable_variables
layer_metrics
U	variables
metrics
layers
 layer_regularization_losses
Vregularization_losses
non_trainable_variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ctrainable_variables
layer_metrics
d	variables
metrics
layers
 layer_regularization_losses
eregularization_losses
non_trainable_variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
mtrainable_variables
layer_metrics
n	variables
metrics
layers
  layer_regularization_losses
oregularization_losses
Ёnon_trainable_variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

Ђtotal

Ѓcount
Є	variables
Ѕ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ј

Іtotal

Їcount
Ј
_fn_kwargs
Љ	variables
Њ	keras_api"Ќ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
љ

Ћtotal

Ќcount
­
_fn_kwargs
Ў	variables
Џ	keras_api"­
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Ђ0
Ѓ1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
І0
Ї1"
trackable_list_wrapper
.
Љ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
.
Ў	variables"
_generic_user_object
':%2 2Adam/dense_327/kernel/m
!: 2Adam/dense_327/bias/m
':%2@2Adam/dense_328/kernel/m
!:@2Adam/dense_328/bias/m
':%`2Adam/dense_329/kernel/m
!:2Adam/dense_329/bias/m
3:1	Ќ2"Adam/lstm_68/lstm_cell_68/kernel/m
=:;	KЌ2,Adam/lstm_68/lstm_cell_68/recurrent_kernel/m
-:+Ќ2 Adam/lstm_68/lstm_cell_68/bias/m
1:/	2 Adam/gru_59/gru_cell_59/kernel/m
;:9	22*Adam/gru_59/gru_cell_59/recurrent_kernel/m
/:-	2Adam/gru_59/gru_cell_59/bias/m
3:1	KШ2"Adam/lstm_69/lstm_cell_69/kernel/m
=:;	2Ш2,Adam/lstm_69/lstm_cell_69/recurrent_kernel/m
-:+Ш2 Adam/lstm_69/lstm_cell_69/bias/m
':%2 2Adam/dense_327/kernel/v
!: 2Adam/dense_327/bias/v
':%2@2Adam/dense_328/kernel/v
!:@2Adam/dense_328/bias/v
':%`2Adam/dense_329/kernel/v
!:2Adam/dense_329/bias/v
3:1	Ќ2"Adam/lstm_68/lstm_cell_68/kernel/v
=:;	KЌ2,Adam/lstm_68/lstm_cell_68/recurrent_kernel/v
-:+Ќ2 Adam/lstm_68/lstm_cell_68/bias/v
1:/	2 Adam/gru_59/gru_cell_59/kernel/v
;:9	22*Adam/gru_59/gru_cell_59/recurrent_kernel/v
/:-	2Adam/gru_59/gru_cell_59/bias/v
3:1	KШ2"Adam/lstm_69/lstm_cell_69/kernel/v
=:;	2Ш2,Adam/lstm_69/lstm_cell_69/recurrent_kernel/v
-:+Ш2 Adam/lstm_69/lstm_cell_69/bias/v
т2п
E__inference_model_9_layer_call_and_return_conditional_losses_50187266
E__inference_model_9_layer_call_and_return_conditional_losses_50188487
E__inference_model_9_layer_call_and_return_conditional_losses_50187309
E__inference_model_9_layer_call_and_return_conditional_losses_50188006Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
#__inference__wrapped_model_50184306Ф
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *4Ђ1
/,
input_10џџџџџџџџџџџџџџџџџџ
і2ѓ
*__inference_model_9_layer_call_fn_50187388
*__inference_model_9_layer_call_fn_50188557
*__inference_model_9_layer_call_fn_50187466
*__inference_model_9_layer_call_fn_50188522Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188710
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189038
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188863
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189191е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_lstm_68_layer_call_fn_50189202
*__inference_lstm_68_layer_call_fn_50189213
*__inference_lstm_68_layer_call_fn_50188874
*__inference_lstm_68_layer_call_fn_50188885е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189230
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189225Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_dropout_68_layer_call_fn_50189235
-__inference_dropout_68_layer_call_fn_50189240Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ѓ2№
D__inference_gru_59_layer_call_and_return_conditional_losses_50189558
D__inference_gru_59_layer_call_and_return_conditional_losses_50189399
D__inference_gru_59_layer_call_and_return_conditional_losses_50189898
D__inference_gru_59_layer_call_and_return_conditional_losses_50189739е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_gru_59_layer_call_fn_50189920
)__inference_gru_59_layer_call_fn_50189569
)__inference_gru_59_layer_call_fn_50189580
)__inference_gru_59_layer_call_fn_50189909е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190401
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190073
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190554
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190226е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_lstm_69_layer_call_fn_50190576
*__inference_lstm_69_layer_call_fn_50190237
*__inference_lstm_69_layer_call_fn_50190565
*__inference_lstm_69_layer_call_fn_50190248е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190593
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190588Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_dropout_69_layer_call_fn_50190603
-__inference_dropout_69_layer_call_fn_50190598Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ё2ю
G__inference_dense_327_layer_call_and_return_conditional_losses_50190614Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_dense_327_layer_call_fn_50190623Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_328_layer_call_and_return_conditional_losses_50190634Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_dense_328_layer_call_fn_50190643Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_concatenate_9_layer_call_and_return_conditional_losses_50190650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_concatenate_9_layer_call_fn_50190656Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_329_layer_call_and_return_conditional_losses_50190666Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_dense_329_layer_call_fn_50190675Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЮBЫ
&__inference_signature_wrapper_50187511input_10"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190708
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190741О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_68_layer_call_fn_50190758
/__inference_lstm_cell_68_layer_call_fn_50190775О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190815
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190855О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
.__inference_gru_cell_59_layer_call_fn_50190869
.__inference_gru_cell_59_layer_call_fn_50190883О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
м2й
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190916
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190949О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_69_layer_call_fn_50190983
/__inference_lstm_cell_69_layer_call_fn_50190966О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 А
#__inference__wrapped_model_50184306FGHKIJLMN+,12;<>Ђ;
4Ђ1
/,
input_10џџџџџџџџџџџџџџџџџџ
Њ "5Њ2
0
	dense_329# 
	dense_329џџџџџџџџџг
K__inference_concatenate_9_layer_call_and_return_conditional_losses_50190650ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ`
 Њ
0__inference_concatenate_9_layer_call_fn_50190656vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ@
Њ "џџџџџџџџџ`Ї
G__inference_dense_327_layer_call_and_return_conditional_losses_50190614\+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ 
 
,__inference_dense_327_layer_call_fn_50190623O+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ Ї
G__inference_dense_328_layer_call_and_return_conditional_losses_50190634\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ@
 
,__inference_dense_328_layer_call_fn_50190643O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ@Ї
G__inference_dense_329_layer_call_and_return_conditional_losses_50190666\;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dense_329_layer_call_fn_50190675O;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ "џџџџџџџџџТ
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189225v@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Т
H__inference_dropout_68_layer_call_and_return_conditional_losses_50189230v@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 
-__inference_dropout_68_layer_call_fn_50189235i@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p
Њ "%"џџџџџџџџџџџџџџџџџџK
-__inference_dropout_68_layer_call_fn_50189240i@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p 
Њ "%"џџџџџџџџџџџџџџџџџџKЈ
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190588\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "%Ђ"

0џџџџџџџџџ2
 Ј
H__inference_dropout_69_layer_call_and_return_conditional_losses_50190593\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "%Ђ"

0џџџџџџџџџ2
 
-__inference_dropout_69_layer_call_fn_50190598O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "џџџџџџџџџ2
-__inference_dropout_69_layer_call_fn_50190603O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "џџџџџџџџџ2О
D__inference_gru_59_layer_call_and_return_conditional_losses_50189399vKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 О
D__inference_gru_59_layer_call_and_return_conditional_losses_50189558vKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Х
D__inference_gru_59_layer_call_and_return_conditional_losses_50189739}KIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Х
D__inference_gru_59_layer_call_and_return_conditional_losses_50189898}KIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 
)__inference_gru_59_layer_call_fn_50189569iKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
)__inference_gru_59_layer_call_fn_50189580iKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2
)__inference_gru_59_layer_call_fn_50189909pKIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
)__inference_gru_59_layer_call_fn_50189920pKIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190815ЗKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 
I__inference_gru_cell_59_layer_call_and_return_conditional_losses_50190855ЗKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 м
.__inference_gru_cell_59_layer_call_fn_50190869ЉKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2м
.__inference_gru_cell_59_layer_call_fn_50190883ЉKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2д
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188710FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 д
E__inference_lstm_68_layer_call_and_return_conditional_losses_50188863FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Э
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189038FGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Э
E__inference_lstm_68_layer_call_and_return_conditional_losses_50189191FGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Ћ
*__inference_lstm_68_layer_call_fn_50188874}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџKЋ
*__inference_lstm_68_layer_call_fn_50188885}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџKЄ
*__inference_lstm_68_layer_call_fn_50189202vFGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџKЄ
*__inference_lstm_68_layer_call_fn_50189213vFGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџKЦ
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190073}LMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ц
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190226}LMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 П
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190401vLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 П
E__inference_lstm_69_layer_call_and_return_conditional_losses_50190554vLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 
*__inference_lstm_69_layer_call_fn_50190237pLMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p

 
Њ "џџџџџџџџџ2
*__inference_lstm_69_layer_call_fn_50190248pLMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "џџџџџџџџџ2
*__inference_lstm_69_layer_call_fn_50190565iLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p

 
Њ "џџџџџџџџџ2
*__inference_lstm_69_layer_call_fn_50190576iLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "џџџџџџџџџ2Ь
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190708§FGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p
Њ "sЂp
iЂf

0/0џџџџџџџџџK
EB

0/1/0џџџџџџџџџK

0/1/1џџџџџџџџџK
 Ь
J__inference_lstm_cell_68_layer_call_and_return_conditional_losses_50190741§FGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџK
EB

0/1/0џџџџџџџџџK

0/1/1џџџџџџџџџK
 Ё
/__inference_lstm_cell_68_layer_call_fn_50190758эFGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p
Њ "cЂ`

0џџџџџџџџџK
A>

1/0џџџџџџџџџK

1/1џџџџџџџџџKЁ
/__inference_lstm_cell_68_layer_call_fn_50190775эFGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p 
Њ "cЂ`

0џџџџџџџџџK
A>

1/0џџџџџџџџџK

1/1џџџџџџџџџKЬ
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190916§LMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ь
J__inference_lstm_cell_69_layer_call_and_return_conditional_losses_50190949§LMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ё
/__inference_lstm_cell_69_layer_call_fn_50190966эLMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ё
/__inference_lstm_cell_69_layer_call_fn_50190983эLMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ъ
E__inference_model_9_layer_call_and_return_conditional_losses_50187266FGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ъ
E__inference_model_9_layer_call_and_return_conditional_losses_50187309FGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ч
E__inference_model_9_layer_call_and_return_conditional_losses_50188006~FGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ч
E__inference_model_9_layer_call_and_return_conditional_losses_50188487~FGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ё
*__inference_model_9_layer_call_fn_50187388sFGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџЁ
*__inference_model_9_layer_call_fn_50187466sFGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
*__inference_model_9_layer_call_fn_50188522qFGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
*__inference_model_9_layer_call_fn_50188557qFGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџП
&__inference_signature_wrapper_50187511FGHKIJLMN+,12;<JЂG
Ђ 
@Њ=
;
input_10/,
input_10џџџџџџџџџџџџџџџџџџ"5Њ2
0
	dense_329# 
	dense_329џџџџџџџџџ