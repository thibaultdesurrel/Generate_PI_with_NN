??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(?
?
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
f
gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namegamma
_
gamma/Read/ReadVariableOpReadVariableOpgamma*
_output_shapes

:2*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2#* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:2#*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:#*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:#*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:
*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:
*
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2* 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

:
2*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:2*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d* 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:2d*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:d*
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	d?*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:?*
dtype0
|
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_33/kernel
u
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel* 
_output_shapes
:
??*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:?*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
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
v
Nadam/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameNadam/gamma/m
o
!Nadam/gamma/m/Read/ReadVariableOpReadVariableOpNadam/gamma/m*
_output_shapes

:2*
dtype0
?
Nadam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2#*(
shared_nameNadam/dense_27/kernel/m
?
+Nadam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_27/kernel/m*
_output_shapes

:2#*
dtype0
?
Nadam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*&
shared_nameNadam/dense_27/bias/m
{
)Nadam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_27/bias/m*
_output_shapes
:#*
dtype0
?
Nadam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*(
shared_nameNadam/dense_28/kernel/m
?
+Nadam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_28/kernel/m*
_output_shapes

:#*
dtype0
?
Nadam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_28/bias/m
{
)Nadam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_28/bias/m*
_output_shapes
:*
dtype0
?
Nadam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_29/kernel/m
?
+Nadam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_29/kernel/m*
_output_shapes

:
*
dtype0
?
Nadam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_29/bias/m
{
)Nadam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_29/bias/m*
_output_shapes
:
*
dtype0
?
Nadam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*(
shared_nameNadam/dense_30/kernel/m
?
+Nadam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_30/kernel/m*
_output_shapes

:
2*
dtype0
?
Nadam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameNadam/dense_30/bias/m
{
)Nadam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_30/bias/m*
_output_shapes
:2*
dtype0
?
Nadam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*(
shared_nameNadam/dense_31/kernel/m
?
+Nadam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_31/kernel/m*
_output_shapes

:2d*
dtype0
?
Nadam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_31/bias/m
{
)Nadam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_31/bias/m*
_output_shapes
:d*
dtype0
?
Nadam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameNadam/dense_32/kernel/m
?
+Nadam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_32/kernel/m*
_output_shapes
:	d?*
dtype0
?
Nadam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_32/bias/m
|
)Nadam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_32/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_33/kernel/m
?
+Nadam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_33/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_33/bias/m
|
)Nadam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_33/bias/m*
_output_shapes	
:?*
dtype0
v
Nadam/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameNadam/gamma/v
o
!Nadam/gamma/v/Read/ReadVariableOpReadVariableOpNadam/gamma/v*
_output_shapes

:2*
dtype0
?
Nadam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2#*(
shared_nameNadam/dense_27/kernel/v
?
+Nadam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_27/kernel/v*
_output_shapes

:2#*
dtype0
?
Nadam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*&
shared_nameNadam/dense_27/bias/v
{
)Nadam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_27/bias/v*
_output_shapes
:#*
dtype0
?
Nadam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*(
shared_nameNadam/dense_28/kernel/v
?
+Nadam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_28/kernel/v*
_output_shapes

:#*
dtype0
?
Nadam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_28/bias/v
{
)Nadam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_28/bias/v*
_output_shapes
:*
dtype0
?
Nadam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_29/kernel/v
?
+Nadam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_29/kernel/v*
_output_shapes

:
*
dtype0
?
Nadam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_29/bias/v
{
)Nadam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_29/bias/v*
_output_shapes
:
*
dtype0
?
Nadam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*(
shared_nameNadam/dense_30/kernel/v
?
+Nadam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_30/kernel/v*
_output_shapes

:
2*
dtype0
?
Nadam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameNadam/dense_30/bias/v
{
)Nadam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_30/bias/v*
_output_shapes
:2*
dtype0
?
Nadam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*(
shared_nameNadam/dense_31/kernel/v
?
+Nadam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_31/kernel/v*
_output_shapes

:2d*
dtype0
?
Nadam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_31/bias/v
{
)Nadam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_31/bias/v*
_output_shapes
:d*
dtype0
?
Nadam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameNadam/dense_32/kernel/v
?
+Nadam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_32/kernel/v*
_output_shapes
:	d?*
dtype0
?
Nadam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_32/bias/v
|
)Nadam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_32/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_33/kernel/v
?
+Nadam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_33/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_33/bias/v
|
)Nadam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_33/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
]
	gamma
trainable_variables
	variables
regularization_losses
	keras_api

	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
h

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
?
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_rate
Kmomentum_cachem?m?m?m?m?$m?%m?.m?/m?4m?5m?:m?;m?@m?Am?v?v?v?v?v?$v?%v?.v?/v?4v?5v?:v?;v?@v?Av?
n
0
1
2
3
4
$5
%6
.7
/8
49
510
:11
;12
@13
A14
n
0
1
2
3
4
$5
%6
.7
/8
49
510
:11
;12
@13
A14
 
?
Lnon_trainable_variables
trainable_variables
Mlayer_metrics
	variables

Nlayers
Olayer_regularization_losses
Pmetrics
regularization_losses
 
PN
VARIABLE_VALUEgamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
Qlayer_metrics
trainable_variables
Rlayer_regularization_losses
	variables

Slayers
Tnon_trainable_variables
Umetrics
regularization_losses
 
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Vlayer_metrics
trainable_variables
Wlayer_regularization_losses
	variables

Xlayers
Ynon_trainable_variables
Zmetrics
regularization_losses
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
[layer_metrics
 trainable_variables
\layer_regularization_losses
!	variables

]layers
^non_trainable_variables
_metrics
"regularization_losses
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
`layer_metrics
&trainable_variables
alayer_regularization_losses
'	variables

blayers
cnon_trainable_variables
dmetrics
(regularization_losses
 
 
 
?
elayer_metrics
*trainable_variables
flayer_regularization_losses
+	variables

glayers
hnon_trainable_variables
imetrics
,regularization_losses
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
jlayer_metrics
0trainable_variables
klayer_regularization_losses
1	variables

llayers
mnon_trainable_variables
nmetrics
2regularization_losses
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
olayer_metrics
6trainable_variables
player_regularization_losses
7	variables

qlayers
rnon_trainable_variables
smetrics
8regularization_losses
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
tlayer_metrics
<trainable_variables
ulayer_regularization_losses
=	variables

vlayers
wnon_trainable_variables
xmetrics
>regularization_losses
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
?
ylayer_metrics
Btrainable_variables
zlayer_regularization_losses
C	variables

{layers
|non_trainable_variables
}metrics
Dregularization_losses
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 
 
N
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
10
 

~0
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
7
	total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
?1

?	variables
tr
VARIABLE_VALUENadam/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_27/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_27/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_28/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_28/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_29/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_29/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_30/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_30/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_31/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_31/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_32/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_32/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_33/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_33/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUENadam/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_27/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_27/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_28/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_28/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_29/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_29/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_30/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_30/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_31/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_31/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_32/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_32/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_33/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_33/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5gammadense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_33780
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamegamma/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp!Nadam/gamma/m/Read/ReadVariableOp+Nadam/dense_27/kernel/m/Read/ReadVariableOp)Nadam/dense_27/bias/m/Read/ReadVariableOp+Nadam/dense_28/kernel/m/Read/ReadVariableOp)Nadam/dense_28/bias/m/Read/ReadVariableOp+Nadam/dense_29/kernel/m/Read/ReadVariableOp)Nadam/dense_29/bias/m/Read/ReadVariableOp+Nadam/dense_30/kernel/m/Read/ReadVariableOp)Nadam/dense_30/bias/m/Read/ReadVariableOp+Nadam/dense_31/kernel/m/Read/ReadVariableOp)Nadam/dense_31/bias/m/Read/ReadVariableOp+Nadam/dense_32/kernel/m/Read/ReadVariableOp)Nadam/dense_32/bias/m/Read/ReadVariableOp+Nadam/dense_33/kernel/m/Read/ReadVariableOp)Nadam/dense_33/bias/m/Read/ReadVariableOp!Nadam/gamma/v/Read/ReadVariableOp+Nadam/dense_27/kernel/v/Read/ReadVariableOp)Nadam/dense_27/bias/v/Read/ReadVariableOp+Nadam/dense_28/kernel/v/Read/ReadVariableOp)Nadam/dense_28/bias/v/Read/ReadVariableOp+Nadam/dense_29/kernel/v/Read/ReadVariableOp)Nadam/dense_29/bias/v/Read/ReadVariableOp+Nadam/dense_30/kernel/v/Read/ReadVariableOp)Nadam/dense_30/bias/v/Read/ReadVariableOp+Nadam/dense_31/kernel/v/Read/ReadVariableOp)Nadam/dense_31/bias/v/Read/ReadVariableOp+Nadam/dense_32/kernel/v/Read/ReadVariableOp)Nadam/dense_32/bias/v/Read/ReadVariableOp+Nadam/dense_33/kernel/v/Read/ReadVariableOp)Nadam/dense_33/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_34506
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegammadense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/gamma/mNadam/dense_27/kernel/mNadam/dense_27/bias/mNadam/dense_28/kernel/mNadam/dense_28/bias/mNadam/dense_29/kernel/mNadam/dense_29/bias/mNadam/dense_30/kernel/mNadam/dense_30/bias/mNadam/dense_31/kernel/mNadam/dense_31/bias/mNadam/dense_32/kernel/mNadam/dense_32/bias/mNadam/dense_33/kernel/mNadam/dense_33/bias/mNadam/gamma/vNadam/dense_27/kernel/vNadam/dense_27/bias/vNadam/dense_28/kernel/vNadam/dense_28/bias/vNadam/dense_29/kernel/vNadam/dense_29/bias/vNadam/dense_30/kernel/vNadam/dense_30/bias/vNadam/dense_31/kernel/vNadam/dense_31/bias/vNadam/dense_32/kernel/vNadam/dense_32/bias/vNadam/dense_33/kernel/vNadam/dense_33/bias/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_34675о
?
?
'__inference_model_4_layer_call_fn_33850

inputs
unknown:2
	unknown_0:2#
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:

	unknown_5:

	unknown_6:
2
	unknown_7:2
	unknown_8:2d
	unknown_9:d

unknown_10:	d?

unknown_11:	?

unknown_12:
??

unknown_13:	?
identity??StatefulPartitionedCall?
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
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_335812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
B__inference_model_4_layer_call_and_return_conditional_losses_33737
input_5*
fully_connected2_4_33696:2 
dense_27_33700:2#
dense_27_33702:# 
dense_28_33705:#
dense_28_33707: 
dense_29_33710:

dense_29_33712:
 
dense_30_33716:
2
dense_30_33718:2 
dense_31_33721:2d
dense_31_33723:d!
dense_32_33726:	d?
dense_32_33728:	?"
dense_33_33731:
??
dense_33_33733:	?
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?*fully_connected2_4/StatefulPartitionedCall?
*fully_connected2_4/StatefulPartitionedCallStatefulPartitionedCallinput_5fully_connected2_4_33696*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_331792,
*fully_connected2_4/StatefulPartitionedCall?
tf.nn.relu_4/ReluRelu3fully_connected2_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_27_33700dense_27_33702*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_332152"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_33705dense_28_33707*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_332522"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_33710dense_29_33712*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_332892"
 dense_29/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_334592
lambda_4/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_30_33716dense_30_33718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_333142"
 dense_30/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_33721dense_31_33723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_333312"
 dense_31/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_33726dense_32_33728*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_333482"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_33731dense_33_33733*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_333652"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall+^fully_connected2_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2X
*fully_connected2_4/StatefulPartitionedCall*fully_connected2_4/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
?
?
(__inference_dense_28_layer_call_fn_34151

inputs
unknown:#
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_332522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????#: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs
?
?
(__inference_dense_27_layer_call_fn_34111

inputs
unknown:2#
	unknown_0:#
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_332152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_33780
input_5
unknown:2
	unknown_0:2#
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:

	unknown_5:

	unknown_6:
2
	unknown_7:2
	unknown_8:2d
	unknown_9:d

unknown_10:	d?

unknown_11:	?

unknown_12:
??

unknown_13:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_331652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
?
?
(__inference_dense_30_layer_call_fn_34253

inputs
unknown:
2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_333142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
? 
?
C__inference_dense_28_layer_call_and_return_conditional_losses_33252

inputs3
!tensordot_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????#2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs
?
?
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_33179

inputs7
%einsum_einsum_readvariableop_resource:2
identity??einsum/Einsum/ReadVariableOp?
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:2*
dtype02
einsum/Einsum/ReadVariableOp?
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????2*
equationijk,kl->ijl2
einsum/Einsum?
IdentityIdentityeinsum/Einsum:output:0^einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:??????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_32_layer_call_and_return_conditional_losses_34304

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
C__inference_dense_33_layer_call_and_return_conditional_losses_33365

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_33331

inputs0
matmul_readvariableop_resource:2d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_dense_31_layer_call_fn_34273

inputs
unknown:2d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_333312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_dense_32_layer_call_fn_34293

inputs
unknown:	d?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_333482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
(__inference_dense_33_layer_call_fn_34313

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_333652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?k
?
__inference__traced_save_34506
file_prefix$
 savev2_gamma_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop,
(savev2_nadam_gamma_m_read_readvariableop6
2savev2_nadam_dense_27_kernel_m_read_readvariableop4
0savev2_nadam_dense_27_bias_m_read_readvariableop6
2savev2_nadam_dense_28_kernel_m_read_readvariableop4
0savev2_nadam_dense_28_bias_m_read_readvariableop6
2savev2_nadam_dense_29_kernel_m_read_readvariableop4
0savev2_nadam_dense_29_bias_m_read_readvariableop6
2savev2_nadam_dense_30_kernel_m_read_readvariableop4
0savev2_nadam_dense_30_bias_m_read_readvariableop6
2savev2_nadam_dense_31_kernel_m_read_readvariableop4
0savev2_nadam_dense_31_bias_m_read_readvariableop6
2savev2_nadam_dense_32_kernel_m_read_readvariableop4
0savev2_nadam_dense_32_bias_m_read_readvariableop6
2savev2_nadam_dense_33_kernel_m_read_readvariableop4
0savev2_nadam_dense_33_bias_m_read_readvariableop,
(savev2_nadam_gamma_v_read_readvariableop6
2savev2_nadam_dense_27_kernel_v_read_readvariableop4
0savev2_nadam_dense_27_bias_v_read_readvariableop6
2savev2_nadam_dense_28_kernel_v_read_readvariableop4
0savev2_nadam_dense_28_bias_v_read_readvariableop6
2savev2_nadam_dense_29_kernel_v_read_readvariableop4
0savev2_nadam_dense_29_bias_v_read_readvariableop6
2savev2_nadam_dense_30_kernel_v_read_readvariableop4
0savev2_nadam_dense_30_bias_v_read_readvariableop6
2savev2_nadam_dense_31_kernel_v_read_readvariableop4
0savev2_nadam_dense_31_bias_v_read_readvariableop6
2savev2_nadam_dense_32_kernel_v_read_readvariableop4
0savev2_nadam_dense_32_bias_v_read_readvariableop6
2savev2_nadam_dense_33_kernel_v_read_readvariableop4
0savev2_nadam_dense_33_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_gamma_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop(savev2_nadam_gamma_m_read_readvariableop2savev2_nadam_dense_27_kernel_m_read_readvariableop0savev2_nadam_dense_27_bias_m_read_readvariableop2savev2_nadam_dense_28_kernel_m_read_readvariableop0savev2_nadam_dense_28_bias_m_read_readvariableop2savev2_nadam_dense_29_kernel_m_read_readvariableop0savev2_nadam_dense_29_bias_m_read_readvariableop2savev2_nadam_dense_30_kernel_m_read_readvariableop0savev2_nadam_dense_30_bias_m_read_readvariableop2savev2_nadam_dense_31_kernel_m_read_readvariableop0savev2_nadam_dense_31_bias_m_read_readvariableop2savev2_nadam_dense_32_kernel_m_read_readvariableop0savev2_nadam_dense_32_bias_m_read_readvariableop2savev2_nadam_dense_33_kernel_m_read_readvariableop0savev2_nadam_dense_33_bias_m_read_readvariableop(savev2_nadam_gamma_v_read_readvariableop2savev2_nadam_dense_27_kernel_v_read_readvariableop0savev2_nadam_dense_27_bias_v_read_readvariableop2savev2_nadam_dense_28_kernel_v_read_readvariableop0savev2_nadam_dense_28_bias_v_read_readvariableop2savev2_nadam_dense_29_kernel_v_read_readvariableop0savev2_nadam_dense_29_bias_v_read_readvariableop2savev2_nadam_dense_30_kernel_v_read_readvariableop0savev2_nadam_dense_30_bias_v_read_readvariableop2savev2_nadam_dense_31_kernel_v_read_readvariableop0savev2_nadam_dense_31_bias_v_read_readvariableop2savev2_nadam_dense_32_kernel_v_read_readvariableop0savev2_nadam_dense_32_bias_v_read_readvariableop2savev2_nadam_dense_33_kernel_v_read_readvariableop0savev2_nadam_dense_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:2#:#:#::
:
:
2:2:2d:d:	d?:?:
??:?: : : : : : : : :2:2#:#:#::
:
:
2:2:2d:d:	d?:?:
??:?:2:2#:#:#::
:
:
2:2:2d:d:	d?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2:$ 

_output_shapes

:2#: 

_output_shapes
:#:$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
2: 	

_output_shapes
:2:$
 

_output_shapes

:2d: 

_output_shapes
:d:%!

_output_shapes
:	d?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2:$ 

_output_shapes

:2#: 

_output_shapes
:#:$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
2:  

_output_shapes
:2:$! 

_output_shapes

:2d: "

_output_shapes
:d:%#!

_output_shapes
:	d?:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:$' 

_output_shapes

:2:$( 

_output_shapes

:2#: )

_output_shapes
:#:$* 

_output_shapes

:#: +

_output_shapes
::$, 

_output_shapes

:
: -

_output_shapes
:
:$. 

_output_shapes

:
2: /

_output_shapes
:2:$0 

_output_shapes

:2d: 1

_output_shapes
:d:%2!

_output_shapes
:	d?:!3

_output_shapes	
:?:&4"
 
_output_shapes
:
??:!5

_output_shapes	
:?:6

_output_shapes
: 
? 
?
C__inference_dense_28_layer_call_and_return_conditional_losses_34182

inputs3
!tensordot_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????#2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????#
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_33165
input_5R
@model_4_fully_connected2_4_einsum_einsum_readvariableop_resource:2D
2model_4_dense_27_tensordot_readvariableop_resource:2#>
0model_4_dense_27_biasadd_readvariableop_resource:#D
2model_4_dense_28_tensordot_readvariableop_resource:#>
0model_4_dense_28_biasadd_readvariableop_resource:D
2model_4_dense_29_tensordot_readvariableop_resource:
>
0model_4_dense_29_biasadd_readvariableop_resource:
A
/model_4_dense_30_matmul_readvariableop_resource:
2>
0model_4_dense_30_biasadd_readvariableop_resource:2A
/model_4_dense_31_matmul_readvariableop_resource:2d>
0model_4_dense_31_biasadd_readvariableop_resource:dB
/model_4_dense_32_matmul_readvariableop_resource:	d??
0model_4_dense_32_biasadd_readvariableop_resource:	?C
/model_4_dense_33_matmul_readvariableop_resource:
???
0model_4_dense_33_biasadd_readvariableop_resource:	?
identity??'model_4/dense_27/BiasAdd/ReadVariableOp?)model_4/dense_27/Tensordot/ReadVariableOp?'model_4/dense_28/BiasAdd/ReadVariableOp?)model_4/dense_28/Tensordot/ReadVariableOp?'model_4/dense_29/BiasAdd/ReadVariableOp?)model_4/dense_29/Tensordot/ReadVariableOp?'model_4/dense_30/BiasAdd/ReadVariableOp?&model_4/dense_30/MatMul/ReadVariableOp?'model_4/dense_31/BiasAdd/ReadVariableOp?&model_4/dense_31/MatMul/ReadVariableOp?'model_4/dense_32/BiasAdd/ReadVariableOp?&model_4/dense_32/MatMul/ReadVariableOp?'model_4/dense_33/BiasAdd/ReadVariableOp?&model_4/dense_33/MatMul/ReadVariableOp?7model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp?
7model_4/fully_connected2_4/einsum/Einsum/ReadVariableOpReadVariableOp@model_4_fully_connected2_4_einsum_einsum_readvariableop_resource*
_output_shapes

:2*
dtype029
7model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp?
(model_4/fully_connected2_4/einsum/EinsumEinsuminput_5?model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????2*
equationijk,kl->ijl2*
(model_4/fully_connected2_4/einsum/Einsum?
model_4/tf.nn.relu_4/ReluRelu1model_4/fully_connected2_4/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????22
model_4/tf.nn.relu_4/Relu?
)model_4/dense_27/Tensordot/ReadVariableOpReadVariableOp2model_4_dense_27_tensordot_readvariableop_resource*
_output_shapes

:2#*
dtype02+
)model_4/dense_27/Tensordot/ReadVariableOp?
model_4/dense_27/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
model_4/dense_27/Tensordot/axes?
model_4/dense_27/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
model_4/dense_27/Tensordot/free?
 model_4/dense_27/Tensordot/ShapeShape'model_4/tf.nn.relu_4/Relu:activations:0*
T0*
_output_shapes
:2"
 model_4/dense_27/Tensordot/Shape?
(model_4/dense_27/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_27/Tensordot/GatherV2/axis?
#model_4/dense_27/Tensordot/GatherV2GatherV2)model_4/dense_27/Tensordot/Shape:output:0(model_4/dense_27/Tensordot/free:output:01model_4/dense_27/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model_4/dense_27/Tensordot/GatherV2?
*model_4/dense_27/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_4/dense_27/Tensordot/GatherV2_1/axis?
%model_4/dense_27/Tensordot/GatherV2_1GatherV2)model_4/dense_27/Tensordot/Shape:output:0(model_4/dense_27/Tensordot/axes:output:03model_4/dense_27/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%model_4/dense_27/Tensordot/GatherV2_1?
 model_4/dense_27/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model_4/dense_27/Tensordot/Const?
model_4/dense_27/Tensordot/ProdProd,model_4/dense_27/Tensordot/GatherV2:output:0)model_4/dense_27/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
model_4/dense_27/Tensordot/Prod?
"model_4/dense_27/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model_4/dense_27/Tensordot/Const_1?
!model_4/dense_27/Tensordot/Prod_1Prod.model_4/dense_27/Tensordot/GatherV2_1:output:0+model_4/dense_27/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!model_4/dense_27/Tensordot/Prod_1?
&model_4/dense_27/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/dense_27/Tensordot/concat/axis?
!model_4/dense_27/Tensordot/concatConcatV2(model_4/dense_27/Tensordot/free:output:0(model_4/dense_27/Tensordot/axes:output:0/model_4/dense_27/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!model_4/dense_27/Tensordot/concat?
 model_4/dense_27/Tensordot/stackPack(model_4/dense_27/Tensordot/Prod:output:0*model_4/dense_27/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 model_4/dense_27/Tensordot/stack?
$model_4/dense_27/Tensordot/transpose	Transpose'model_4/tf.nn.relu_4/Relu:activations:0*model_4/dense_27/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????22&
$model_4/dense_27/Tensordot/transpose?
"model_4/dense_27/Tensordot/ReshapeReshape(model_4/dense_27/Tensordot/transpose:y:0)model_4/dense_27/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"model_4/dense_27/Tensordot/Reshape?
!model_4/dense_27/Tensordot/MatMulMatMul+model_4/dense_27/Tensordot/Reshape:output:01model_4/dense_27/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????#2#
!model_4/dense_27/Tensordot/MatMul?
"model_4/dense_27/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2$
"model_4/dense_27/Tensordot/Const_2?
(model_4/dense_27/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_27/Tensordot/concat_1/axis?
#model_4/dense_27/Tensordot/concat_1ConcatV2,model_4/dense_27/Tensordot/GatherV2:output:0+model_4/dense_27/Tensordot/Const_2:output:01model_4/dense_27/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_4/dense_27/Tensordot/concat_1?
model_4/dense_27/TensordotReshape+model_4/dense_27/Tensordot/MatMul:product:0,model_4/dense_27/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????#2
model_4/dense_27/Tensordot?
'model_4/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_27_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02)
'model_4/dense_27/BiasAdd/ReadVariableOp?
model_4/dense_27/BiasAddBiasAdd#model_4/dense_27/Tensordot:output:0/model_4/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????#2
model_4/dense_27/BiasAdd?
model_4/dense_27/ReluRelu!model_4/dense_27/BiasAdd:output:0*
T0*,
_output_shapes
:??????????#2
model_4/dense_27/Relu?
)model_4/dense_28/Tensordot/ReadVariableOpReadVariableOp2model_4_dense_28_tensordot_readvariableop_resource*
_output_shapes

:#*
dtype02+
)model_4/dense_28/Tensordot/ReadVariableOp?
model_4/dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
model_4/dense_28/Tensordot/axes?
model_4/dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
model_4/dense_28/Tensordot/free?
 model_4/dense_28/Tensordot/ShapeShape#model_4/dense_27/Relu:activations:0*
T0*
_output_shapes
:2"
 model_4/dense_28/Tensordot/Shape?
(model_4/dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_28/Tensordot/GatherV2/axis?
#model_4/dense_28/Tensordot/GatherV2GatherV2)model_4/dense_28/Tensordot/Shape:output:0(model_4/dense_28/Tensordot/free:output:01model_4/dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model_4/dense_28/Tensordot/GatherV2?
*model_4/dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_4/dense_28/Tensordot/GatherV2_1/axis?
%model_4/dense_28/Tensordot/GatherV2_1GatherV2)model_4/dense_28/Tensordot/Shape:output:0(model_4/dense_28/Tensordot/axes:output:03model_4/dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%model_4/dense_28/Tensordot/GatherV2_1?
 model_4/dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model_4/dense_28/Tensordot/Const?
model_4/dense_28/Tensordot/ProdProd,model_4/dense_28/Tensordot/GatherV2:output:0)model_4/dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
model_4/dense_28/Tensordot/Prod?
"model_4/dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model_4/dense_28/Tensordot/Const_1?
!model_4/dense_28/Tensordot/Prod_1Prod.model_4/dense_28/Tensordot/GatherV2_1:output:0+model_4/dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!model_4/dense_28/Tensordot/Prod_1?
&model_4/dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/dense_28/Tensordot/concat/axis?
!model_4/dense_28/Tensordot/concatConcatV2(model_4/dense_28/Tensordot/free:output:0(model_4/dense_28/Tensordot/axes:output:0/model_4/dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!model_4/dense_28/Tensordot/concat?
 model_4/dense_28/Tensordot/stackPack(model_4/dense_28/Tensordot/Prod:output:0*model_4/dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 model_4/dense_28/Tensordot/stack?
$model_4/dense_28/Tensordot/transpose	Transpose#model_4/dense_27/Relu:activations:0*model_4/dense_28/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????#2&
$model_4/dense_28/Tensordot/transpose?
"model_4/dense_28/Tensordot/ReshapeReshape(model_4/dense_28/Tensordot/transpose:y:0)model_4/dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"model_4/dense_28/Tensordot/Reshape?
!model_4/dense_28/Tensordot/MatMulMatMul+model_4/dense_28/Tensordot/Reshape:output:01model_4/dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_4/dense_28/Tensordot/MatMul?
"model_4/dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model_4/dense_28/Tensordot/Const_2?
(model_4/dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_28/Tensordot/concat_1/axis?
#model_4/dense_28/Tensordot/concat_1ConcatV2,model_4/dense_28/Tensordot/GatherV2:output:0+model_4/dense_28/Tensordot/Const_2:output:01model_4/dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_4/dense_28/Tensordot/concat_1?
model_4/dense_28/TensordotReshape+model_4/dense_28/Tensordot/MatMul:product:0,model_4/dense_28/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
model_4/dense_28/Tensordot?
'model_4/dense_28/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_28/BiasAdd/ReadVariableOp?
model_4/dense_28/BiasAddBiasAdd#model_4/dense_28/Tensordot:output:0/model_4/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_4/dense_28/BiasAdd?
model_4/dense_28/ReluRelu!model_4/dense_28/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_4/dense_28/Relu?
)model_4/dense_29/Tensordot/ReadVariableOpReadVariableOp2model_4_dense_29_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02+
)model_4/dense_29/Tensordot/ReadVariableOp?
model_4/dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
model_4/dense_29/Tensordot/axes?
model_4/dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
model_4/dense_29/Tensordot/free?
 model_4/dense_29/Tensordot/ShapeShape#model_4/dense_28/Relu:activations:0*
T0*
_output_shapes
:2"
 model_4/dense_29/Tensordot/Shape?
(model_4/dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_29/Tensordot/GatherV2/axis?
#model_4/dense_29/Tensordot/GatherV2GatherV2)model_4/dense_29/Tensordot/Shape:output:0(model_4/dense_29/Tensordot/free:output:01model_4/dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model_4/dense_29/Tensordot/GatherV2?
*model_4/dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_4/dense_29/Tensordot/GatherV2_1/axis?
%model_4/dense_29/Tensordot/GatherV2_1GatherV2)model_4/dense_29/Tensordot/Shape:output:0(model_4/dense_29/Tensordot/axes:output:03model_4/dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%model_4/dense_29/Tensordot/GatherV2_1?
 model_4/dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model_4/dense_29/Tensordot/Const?
model_4/dense_29/Tensordot/ProdProd,model_4/dense_29/Tensordot/GatherV2:output:0)model_4/dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
model_4/dense_29/Tensordot/Prod?
"model_4/dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model_4/dense_29/Tensordot/Const_1?
!model_4/dense_29/Tensordot/Prod_1Prod.model_4/dense_29/Tensordot/GatherV2_1:output:0+model_4/dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!model_4/dense_29/Tensordot/Prod_1?
&model_4/dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/dense_29/Tensordot/concat/axis?
!model_4/dense_29/Tensordot/concatConcatV2(model_4/dense_29/Tensordot/free:output:0(model_4/dense_29/Tensordot/axes:output:0/model_4/dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!model_4/dense_29/Tensordot/concat?
 model_4/dense_29/Tensordot/stackPack(model_4/dense_29/Tensordot/Prod:output:0*model_4/dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 model_4/dense_29/Tensordot/stack?
$model_4/dense_29/Tensordot/transpose	Transpose#model_4/dense_28/Relu:activations:0*model_4/dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2&
$model_4/dense_29/Tensordot/transpose?
"model_4/dense_29/Tensordot/ReshapeReshape(model_4/dense_29/Tensordot/transpose:y:0)model_4/dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"model_4/dense_29/Tensordot/Reshape?
!model_4/dense_29/Tensordot/MatMulMatMul+model_4/dense_29/Tensordot/Reshape:output:01model_4/dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!model_4/dense_29/Tensordot/MatMul?
"model_4/dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2$
"model_4/dense_29/Tensordot/Const_2?
(model_4/dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/dense_29/Tensordot/concat_1/axis?
#model_4/dense_29/Tensordot/concat_1ConcatV2,model_4/dense_29/Tensordot/GatherV2:output:0+model_4/dense_29/Tensordot/Const_2:output:01model_4/dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_4/dense_29/Tensordot/concat_1?
model_4/dense_29/TensordotReshape+model_4/dense_29/Tensordot/MatMul:product:0,model_4/dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
model_4/dense_29/Tensordot?
'model_4/dense_29/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_4/dense_29/BiasAdd/ReadVariableOp?
model_4/dense_29/BiasAddBiasAdd#model_4/dense_29/Tensordot:output:0/model_4/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
model_4/dense_29/BiasAdd?
model_4/dense_29/ReluRelu!model_4/dense_29/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
model_4/dense_29/Relu?
&model_4/lambda_4/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_4/lambda_4/Min/reduction_indices?
model_4/lambda_4/MinMin#model_4/dense_29/Relu:activations:0/model_4/lambda_4/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
model_4/lambda_4/Min?
&model_4/dense_30/MatMul/ReadVariableOpReadVariableOp/model_4_dense_30_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02(
&model_4/dense_30/MatMul/ReadVariableOp?
model_4/dense_30/MatMulMatMulmodel_4/lambda_4/Min:output:0.model_4/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/dense_30/MatMul?
'model_4/dense_30/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'model_4/dense_30/BiasAdd/ReadVariableOp?
model_4/dense_30/BiasAddBiasAdd!model_4/dense_30/MatMul:product:0/model_4/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_4/dense_30/BiasAdd?
model_4/dense_30/ReluRelu!model_4/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_4/dense_30/Relu?
&model_4/dense_31/MatMul/ReadVariableOpReadVariableOp/model_4_dense_31_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02(
&model_4/dense_31/MatMul/ReadVariableOp?
model_4/dense_31/MatMulMatMul#model_4/dense_30/Relu:activations:0.model_4/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_4/dense_31/MatMul?
'model_4/dense_31/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_31_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'model_4/dense_31/BiasAdd/ReadVariableOp?
model_4/dense_31/BiasAddBiasAdd!model_4/dense_31/MatMul:product:0/model_4/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_4/dense_31/BiasAdd?
model_4/dense_31/ReluRelu!model_4/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
model_4/dense_31/Relu?
&model_4/dense_32/MatMul/ReadVariableOpReadVariableOp/model_4_dense_32_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02(
&model_4/dense_32/MatMul/ReadVariableOp?
model_4/dense_32/MatMulMatMul#model_4/dense_31/Relu:activations:0.model_4/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_32/MatMul?
'model_4/dense_32/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_4/dense_32/BiasAdd/ReadVariableOp?
model_4/dense_32/BiasAddBiasAdd!model_4/dense_32/MatMul:product:0/model_4/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_32/BiasAdd?
model_4/dense_32/ReluRelu!model_4/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_32/Relu?
&model_4/dense_33/MatMul/ReadVariableOpReadVariableOp/model_4_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_4/dense_33/MatMul/ReadVariableOp?
model_4/dense_33/MatMulMatMul#model_4/dense_32/Relu:activations:0.model_4/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_33/MatMul?
'model_4/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_4/dense_33/BiasAdd/ReadVariableOp?
model_4/dense_33/BiasAddBiasAdd!model_4/dense_33/MatMul:product:0/model_4/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_33/BiasAdd?
model_4/dense_33/SigmoidSigmoid!model_4/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_33/Sigmoid?
IdentityIdentitymodel_4/dense_33/Sigmoid:y:0(^model_4/dense_27/BiasAdd/ReadVariableOp*^model_4/dense_27/Tensordot/ReadVariableOp(^model_4/dense_28/BiasAdd/ReadVariableOp*^model_4/dense_28/Tensordot/ReadVariableOp(^model_4/dense_29/BiasAdd/ReadVariableOp*^model_4/dense_29/Tensordot/ReadVariableOp(^model_4/dense_30/BiasAdd/ReadVariableOp'^model_4/dense_30/MatMul/ReadVariableOp(^model_4/dense_31/BiasAdd/ReadVariableOp'^model_4/dense_31/MatMul/ReadVariableOp(^model_4/dense_32/BiasAdd/ReadVariableOp'^model_4/dense_32/MatMul/ReadVariableOp(^model_4/dense_33/BiasAdd/ReadVariableOp'^model_4/dense_33/MatMul/ReadVariableOp8^model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2R
'model_4/dense_27/BiasAdd/ReadVariableOp'model_4/dense_27/BiasAdd/ReadVariableOp2V
)model_4/dense_27/Tensordot/ReadVariableOp)model_4/dense_27/Tensordot/ReadVariableOp2R
'model_4/dense_28/BiasAdd/ReadVariableOp'model_4/dense_28/BiasAdd/ReadVariableOp2V
)model_4/dense_28/Tensordot/ReadVariableOp)model_4/dense_28/Tensordot/ReadVariableOp2R
'model_4/dense_29/BiasAdd/ReadVariableOp'model_4/dense_29/BiasAdd/ReadVariableOp2V
)model_4/dense_29/Tensordot/ReadVariableOp)model_4/dense_29/Tensordot/ReadVariableOp2R
'model_4/dense_30/BiasAdd/ReadVariableOp'model_4/dense_30/BiasAdd/ReadVariableOp2P
&model_4/dense_30/MatMul/ReadVariableOp&model_4/dense_30/MatMul/ReadVariableOp2R
'model_4/dense_31/BiasAdd/ReadVariableOp'model_4/dense_31/BiasAdd/ReadVariableOp2P
&model_4/dense_31/MatMul/ReadVariableOp&model_4/dense_31/MatMul/ReadVariableOp2R
'model_4/dense_32/BiasAdd/ReadVariableOp'model_4/dense_32/BiasAdd/ReadVariableOp2P
&model_4/dense_32/MatMul/ReadVariableOp&model_4/dense_32/MatMul/ReadVariableOp2R
'model_4/dense_33/BiasAdd/ReadVariableOp'model_4/dense_33/BiasAdd/ReadVariableOp2P
&model_4/dense_33/MatMul/ReadVariableOp&model_4/dense_33/MatMul/ReadVariableOp2r
7model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp7model_4/fully_connected2_4/einsum/Einsum/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
Ξ
?
B__inference_model_4_layer_call_and_return_conditional_losses_34088

inputsJ
8fully_connected2_4_einsum_einsum_readvariableop_resource:2<
*dense_27_tensordot_readvariableop_resource:2#6
(dense_27_biasadd_readvariableop_resource:#<
*dense_28_tensordot_readvariableop_resource:#6
(dense_28_biasadd_readvariableop_resource:<
*dense_29_tensordot_readvariableop_resource:
6
(dense_29_biasadd_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
26
(dense_30_biasadd_readvariableop_resource:29
'dense_31_matmul_readvariableop_resource:2d6
(dense_31_biasadd_readvariableop_resource:d:
'dense_32_matmul_readvariableop_resource:	d?7
(dense_32_biasadd_readvariableop_resource:	?;
'dense_33_matmul_readvariableop_resource:
??7
(dense_33_biasadd_readvariableop_resource:	?
identity??dense_27/BiasAdd/ReadVariableOp?!dense_27/Tensordot/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?!dense_28/Tensordot/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?!dense_29/Tensordot/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?/fully_connected2_4/einsum/Einsum/ReadVariableOp?
/fully_connected2_4/einsum/Einsum/ReadVariableOpReadVariableOp8fully_connected2_4_einsum_einsum_readvariableop_resource*
_output_shapes

:2*
dtype021
/fully_connected2_4/einsum/Einsum/ReadVariableOp?
 fully_connected2_4/einsum/EinsumEinsuminputs7fully_connected2_4/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????2*
equationijk,kl->ijl2"
 fully_connected2_4/einsum/Einsum?
tf.nn.relu_4/ReluRelu)fully_connected2_4/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
!dense_27/Tensordot/ReadVariableOpReadVariableOp*dense_27_tensordot_readvariableop_resource*
_output_shapes

:2#*
dtype02#
!dense_27/Tensordot/ReadVariableOp|
dense_27/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_27/Tensordot/axes?
dense_27/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_27/Tensordot/free?
dense_27/Tensordot/ShapeShapetf.nn.relu_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_27/Tensordot/Shape?
 dense_27/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_27/Tensordot/GatherV2/axis?
dense_27/Tensordot/GatherV2GatherV2!dense_27/Tensordot/Shape:output:0 dense_27/Tensordot/free:output:0)dense_27/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_27/Tensordot/GatherV2?
"dense_27/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_27/Tensordot/GatherV2_1/axis?
dense_27/Tensordot/GatherV2_1GatherV2!dense_27/Tensordot/Shape:output:0 dense_27/Tensordot/axes:output:0+dense_27/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_27/Tensordot/GatherV2_1~
dense_27/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_27/Tensordot/Const?
dense_27/Tensordot/ProdProd$dense_27/Tensordot/GatherV2:output:0!dense_27/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_27/Tensordot/Prod?
dense_27/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_27/Tensordot/Const_1?
dense_27/Tensordot/Prod_1Prod&dense_27/Tensordot/GatherV2_1:output:0#dense_27/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_27/Tensordot/Prod_1?
dense_27/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_27/Tensordot/concat/axis?
dense_27/Tensordot/concatConcatV2 dense_27/Tensordot/free:output:0 dense_27/Tensordot/axes:output:0'dense_27/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/concat?
dense_27/Tensordot/stackPack dense_27/Tensordot/Prod:output:0"dense_27/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/stack?
dense_27/Tensordot/transpose	Transposetf.nn.relu_4/Relu:activations:0"dense_27/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????22
dense_27/Tensordot/transpose?
dense_27/Tensordot/ReshapeReshape dense_27/Tensordot/transpose:y:0!dense_27/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_27/Tensordot/Reshape?
dense_27/Tensordot/MatMulMatMul#dense_27/Tensordot/Reshape:output:0)dense_27/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????#2
dense_27/Tensordot/MatMul?
dense_27/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_27/Tensordot/Const_2?
 dense_27/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_27/Tensordot/concat_1/axis?
dense_27/Tensordot/concat_1ConcatV2$dense_27/Tensordot/GatherV2:output:0#dense_27/Tensordot/Const_2:output:0)dense_27/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/concat_1?
dense_27/TensordotReshape#dense_27/Tensordot/MatMul:product:0$dense_27/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????#2
dense_27/Tensordot?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/Tensordot:output:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????#2
dense_27/BiasAddx
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*,
_output_shapes
:??????????#2
dense_27/Relu?
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource*
_output_shapes

:#*
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_28/Tensordot/free
dense_28/Tensordot/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposedense_27/Relu:activations:0"dense_28/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????#2
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_28/BiasAddx
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_28/Relu?
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes?
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape?
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axis?
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2?
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis?
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const?
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod?
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1?
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1?
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axis?
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat?
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stack?
dense_29/Tensordot/transpose	Transposedense_28/Relu:activations:0"dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_29/Tensordot/transpose?
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_29/Tensordot/Reshape?
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_29/Tensordot/MatMul?
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
dense_29/Tensordot/Const_2?
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axis?
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1?
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
dense_29/Tensordot?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
dense_29/BiasAddx
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
dense_29/Relu?
lambda_4/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_4/Min/reduction_indices?
lambda_4/MinMindense_29/Relu:activations:0'lambda_4/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
lambda_4/Min?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMullambda_4/Min:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_30/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_31/BiasAdds
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_31/Relu?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMuldense_31/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAddt
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_32/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd}
dense_33/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_33/Sigmoid?
IdentityIdentitydense_33/Sigmoid:y:0 ^dense_27/BiasAdd/ReadVariableOp"^dense_27/Tensordot/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp"^dense_28/Tensordot/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp"^dense_29/Tensordot/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp0^fully_connected2_4/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2F
!dense_27/Tensordot/ReadVariableOp!dense_27/Tensordot/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/Tensordot/ReadVariableOp!dense_28/Tensordot/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2F
!dense_29/Tensordot/ReadVariableOp!dense_29/Tensordot/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2b
/fully_connected2_4/einsum/Einsum/ReadVariableOp/fully_connected2_4/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
C__inference_dense_27_layer_call_and_return_conditional_losses_34142

inputs3
!tensordot_readvariableop_resource:2#-
biasadd_readvariableop_resource:#
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????22
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????#2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????#2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????#2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
'__inference_model_4_layer_call_fn_33649
input_5
unknown:2
	unknown_0:2#
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:

	unknown_5:

	unknown_6:
2
	unknown_7:2
	unknown_8:2d
	unknown_9:d

unknown_10:	d?

unknown_11:	?

unknown_12:
??

unknown_13:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_335812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
?0
?
B__inference_model_4_layer_call_and_return_conditional_losses_33581

inputs*
fully_connected2_4_33540:2 
dense_27_33544:2#
dense_27_33546:# 
dense_28_33549:#
dense_28_33551: 
dense_29_33554:

dense_29_33556:
 
dense_30_33560:
2
dense_30_33562:2 
dense_31_33565:2d
dense_31_33567:d!
dense_32_33570:	d?
dense_32_33572:	?"
dense_33_33575:
??
dense_33_33577:	?
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?*fully_connected2_4/StatefulPartitionedCall?
*fully_connected2_4/StatefulPartitionedCallStatefulPartitionedCallinputsfully_connected2_4_33540*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_331792,
*fully_connected2_4/StatefulPartitionedCall?
tf.nn.relu_4/ReluRelu3fully_connected2_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_27_33544dense_27_33546*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_332152"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_33549dense_28_33551*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_332522"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_33554dense_29_33556*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_332892"
 dense_29/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_334592
lambda_4/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_30_33560dense_30_33562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_333142"
 dense_30/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_33565dense_31_33567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_333312"
 dense_31/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_33570dense_32_33572*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_333482"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_33575dense_33_33577*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_333652"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall+^fully_connected2_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2X
*fully_connected2_4/StatefulPartitionedCall*fully_connected2_4/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_4_layer_call_and_return_conditional_losses_34238

inputs
identityp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min/reduction_indicesk
MinMininputsMin/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
Min`
IdentityIdentityMin:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?

?
C__inference_dense_30_layer_call_and_return_conditional_losses_33314

inputs0
matmul_readvariableop_resource:
2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
? 
?
C__inference_dense_29_layer_call_and_return_conditional_losses_33289

inputs3
!tensordot_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_4_layer_call_fn_33815

inputs
unknown:2
	unknown_0:2#
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:

	unknown_5:

	unknown_6:
2
	unknown_7:2
	unknown_8:2d
	unknown_9:d

unknown_10:	d?

unknown_11:	?

unknown_12:
??

unknown_13:	?
identity??StatefulPartitionedCall?
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
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_333722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_4_layer_call_and_return_conditional_losses_33301

inputs
identityp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min/reduction_indicesk
MinMininputsMin/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
Min`
IdentityIdentityMin:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?0
?
B__inference_model_4_layer_call_and_return_conditional_losses_33693
input_5*
fully_connected2_4_33652:2 
dense_27_33656:2#
dense_27_33658:# 
dense_28_33661:#
dense_28_33663: 
dense_29_33666:

dense_29_33668:
 
dense_30_33672:
2
dense_30_33674:2 
dense_31_33677:2d
dense_31_33679:d!
dense_32_33682:	d?
dense_32_33684:	?"
dense_33_33687:
??
dense_33_33689:	?
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?*fully_connected2_4/StatefulPartitionedCall?
*fully_connected2_4/StatefulPartitionedCallStatefulPartitionedCallinput_5fully_connected2_4_33652*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_331792,
*fully_connected2_4/StatefulPartitionedCall?
tf.nn.relu_4/ReluRelu3fully_connected2_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_27_33656dense_27_33658*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_332152"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_33661dense_28_33663*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_332522"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_33666dense_29_33668*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_332892"
 dense_29/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_333012
lambda_4/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_30_33672dense_30_33674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_333142"
 dense_30/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_33677dense_31_33679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_333312"
 dense_31/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_33682dense_32_33684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_333482"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_33687dense_33_33689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_333652"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall+^fully_connected2_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2X
*fully_connected2_4/StatefulPartitionedCall*fully_connected2_4/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
?
_
C__inference_lambda_4_layer_call_and_return_conditional_losses_34244

inputs
identityp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min/reduction_indicesk
MinMininputsMin/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
Min`
IdentityIdentityMin:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_34284

inputs0
matmul_readvariableop_resource:2d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

?
C__inference_dense_33_layer_call_and_return_conditional_losses_34324

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
? 
!__inference__traced_restore_34675
file_prefix(
assignvariableop_gamma:24
"assignvariableop_1_dense_27_kernel:2#.
 assignvariableop_2_dense_27_bias:#4
"assignvariableop_3_dense_28_kernel:#.
 assignvariableop_4_dense_28_bias:4
"assignvariableop_5_dense_29_kernel:
.
 assignvariableop_6_dense_29_bias:
4
"assignvariableop_7_dense_30_kernel:
2.
 assignvariableop_8_dense_30_bias:24
"assignvariableop_9_dense_31_kernel:2d/
!assignvariableop_10_dense_31_bias:d6
#assignvariableop_11_dense_32_kernel:	d?0
!assignvariableop_12_dense_32_bias:	?7
#assignvariableop_13_dense_33_kernel:
??0
!assignvariableop_14_dense_33_bias:	?(
assignvariableop_15_nadam_iter:	 *
 assignvariableop_16_nadam_beta_1: *
 assignvariableop_17_nadam_beta_2: )
assignvariableop_18_nadam_decay: 1
'assignvariableop_19_nadam_learning_rate: 2
(assignvariableop_20_nadam_momentum_cache: #
assignvariableop_21_total: #
assignvariableop_22_count: 3
!assignvariableop_23_nadam_gamma_m:2=
+assignvariableop_24_nadam_dense_27_kernel_m:2#7
)assignvariableop_25_nadam_dense_27_bias_m:#=
+assignvariableop_26_nadam_dense_28_kernel_m:#7
)assignvariableop_27_nadam_dense_28_bias_m:=
+assignvariableop_28_nadam_dense_29_kernel_m:
7
)assignvariableop_29_nadam_dense_29_bias_m:
=
+assignvariableop_30_nadam_dense_30_kernel_m:
27
)assignvariableop_31_nadam_dense_30_bias_m:2=
+assignvariableop_32_nadam_dense_31_kernel_m:2d7
)assignvariableop_33_nadam_dense_31_bias_m:d>
+assignvariableop_34_nadam_dense_32_kernel_m:	d?8
)assignvariableop_35_nadam_dense_32_bias_m:	??
+assignvariableop_36_nadam_dense_33_kernel_m:
??8
)assignvariableop_37_nadam_dense_33_bias_m:	?3
!assignvariableop_38_nadam_gamma_v:2=
+assignvariableop_39_nadam_dense_27_kernel_v:2#7
)assignvariableop_40_nadam_dense_27_bias_v:#=
+assignvariableop_41_nadam_dense_28_kernel_v:#7
)assignvariableop_42_nadam_dense_28_bias_v:=
+assignvariableop_43_nadam_dense_29_kernel_v:
7
)assignvariableop_44_nadam_dense_29_bias_v:
=
+assignvariableop_45_nadam_dense_30_kernel_v:
27
)assignvariableop_46_nadam_dense_30_bias_v:2=
+assignvariableop_47_nadam_dense_31_kernel_v:2d7
)assignvariableop_48_nadam_dense_31_bias_v:d>
+assignvariableop_49_nadam_dense_32_kernel_v:	d?8
)assignvariableop_50_nadam_dense_32_bias_v:	??
+assignvariableop_51_nadam_dense_33_kernel_v:
??8
)assignvariableop_52_nadam_dense_33_bias_v:	?
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_27_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_27_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_28_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_28_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_29_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_29_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_30_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_30_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_31_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_31_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_32_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_32_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_33_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_33_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_nadam_iterIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_nadam_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_nadam_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_nadam_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_nadam_momentum_cacheIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_nadam_gamma_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_27_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_27_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_nadam_dense_28_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_nadam_dense_28_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_dense_29_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_dense_29_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_30_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_30_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_31_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_31_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_dense_32_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_dense_32_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_dense_33_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_dense_33_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp!assignvariableop_38_nadam_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_nadam_dense_27_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_nadam_dense_27_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_nadam_dense_28_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_nadam_dense_28_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_nadam_dense_29_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_nadam_dense_29_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_nadam_dense_30_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_nadam_dense_30_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_nadam_dense_31_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_nadam_dense_31_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_nadam_dense_32_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_nadam_dense_32_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_nadam_dense_33_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_nadam_dense_33_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53?	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
? 
?
C__inference_dense_27_layer_call_and_return_conditional_losses_33215

inputs3
!tensordot_readvariableop_resource:2#-
biasadd_readvariableop_resource:#
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????22
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????#2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????#2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????#2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
(__inference_dense_29_layer_call_fn_34191

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_332892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
B__inference_model_4_layer_call_and_return_conditional_losses_33372

inputs*
fully_connected2_4_33180:2 
dense_27_33216:2#
dense_27_33218:# 
dense_28_33253:#
dense_28_33255: 
dense_29_33290:

dense_29_33292:
 
dense_30_33315:
2
dense_30_33317:2 
dense_31_33332:2d
dense_31_33334:d!
dense_32_33349:	d?
dense_32_33351:	?"
dense_33_33366:
??
dense_33_33368:	?
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?*fully_connected2_4/StatefulPartitionedCall?
*fully_connected2_4/StatefulPartitionedCallStatefulPartitionedCallinputsfully_connected2_4_33180*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_331792,
*fully_connected2_4/StatefulPartitionedCall?
tf.nn.relu_4/ReluRelu3fully_connected2_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_27_33216dense_27_33218*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_332152"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_33253dense_28_33255*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_332522"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_33290dense_29_33292*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_332892"
 dense_29/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_333012
lambda_4/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_30_33315dense_30_33317*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_333142"
 dense_30/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_33332dense_31_33334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_333312"
 dense_31/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_33349dense_32_33351*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_333482"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_33366dense_33_33368*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_333652"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall+^fully_connected2_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2X
*fully_connected2_4/StatefulPartitionedCall*fully_connected2_4/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_fully_connected2_4_layer_call_fn_34095

inputs
unknown:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_331792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_32_layer_call_and_return_conditional_losses_33348

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ξ
?
B__inference_model_4_layer_call_and_return_conditional_losses_33969

inputsJ
8fully_connected2_4_einsum_einsum_readvariableop_resource:2<
*dense_27_tensordot_readvariableop_resource:2#6
(dense_27_biasadd_readvariableop_resource:#<
*dense_28_tensordot_readvariableop_resource:#6
(dense_28_biasadd_readvariableop_resource:<
*dense_29_tensordot_readvariableop_resource:
6
(dense_29_biasadd_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
26
(dense_30_biasadd_readvariableop_resource:29
'dense_31_matmul_readvariableop_resource:2d6
(dense_31_biasadd_readvariableop_resource:d:
'dense_32_matmul_readvariableop_resource:	d?7
(dense_32_biasadd_readvariableop_resource:	?;
'dense_33_matmul_readvariableop_resource:
??7
(dense_33_biasadd_readvariableop_resource:	?
identity??dense_27/BiasAdd/ReadVariableOp?!dense_27/Tensordot/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?!dense_28/Tensordot/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?!dense_29/Tensordot/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?/fully_connected2_4/einsum/Einsum/ReadVariableOp?
/fully_connected2_4/einsum/Einsum/ReadVariableOpReadVariableOp8fully_connected2_4_einsum_einsum_readvariableop_resource*
_output_shapes

:2*
dtype021
/fully_connected2_4/einsum/Einsum/ReadVariableOp?
 fully_connected2_4/einsum/EinsumEinsuminputs7fully_connected2_4/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????2*
equationijk,kl->ijl2"
 fully_connected2_4/einsum/Einsum?
tf.nn.relu_4/ReluRelu)fully_connected2_4/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????22
tf.nn.relu_4/Relu?
!dense_27/Tensordot/ReadVariableOpReadVariableOp*dense_27_tensordot_readvariableop_resource*
_output_shapes

:2#*
dtype02#
!dense_27/Tensordot/ReadVariableOp|
dense_27/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_27/Tensordot/axes?
dense_27/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_27/Tensordot/free?
dense_27/Tensordot/ShapeShapetf.nn.relu_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_27/Tensordot/Shape?
 dense_27/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_27/Tensordot/GatherV2/axis?
dense_27/Tensordot/GatherV2GatherV2!dense_27/Tensordot/Shape:output:0 dense_27/Tensordot/free:output:0)dense_27/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_27/Tensordot/GatherV2?
"dense_27/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_27/Tensordot/GatherV2_1/axis?
dense_27/Tensordot/GatherV2_1GatherV2!dense_27/Tensordot/Shape:output:0 dense_27/Tensordot/axes:output:0+dense_27/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_27/Tensordot/GatherV2_1~
dense_27/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_27/Tensordot/Const?
dense_27/Tensordot/ProdProd$dense_27/Tensordot/GatherV2:output:0!dense_27/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_27/Tensordot/Prod?
dense_27/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_27/Tensordot/Const_1?
dense_27/Tensordot/Prod_1Prod&dense_27/Tensordot/GatherV2_1:output:0#dense_27/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_27/Tensordot/Prod_1?
dense_27/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_27/Tensordot/concat/axis?
dense_27/Tensordot/concatConcatV2 dense_27/Tensordot/free:output:0 dense_27/Tensordot/axes:output:0'dense_27/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/concat?
dense_27/Tensordot/stackPack dense_27/Tensordot/Prod:output:0"dense_27/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/stack?
dense_27/Tensordot/transpose	Transposetf.nn.relu_4/Relu:activations:0"dense_27/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????22
dense_27/Tensordot/transpose?
dense_27/Tensordot/ReshapeReshape dense_27/Tensordot/transpose:y:0!dense_27/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_27/Tensordot/Reshape?
dense_27/Tensordot/MatMulMatMul#dense_27/Tensordot/Reshape:output:0)dense_27/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????#2
dense_27/Tensordot/MatMul?
dense_27/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_27/Tensordot/Const_2?
 dense_27/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_27/Tensordot/concat_1/axis?
dense_27/Tensordot/concat_1ConcatV2$dense_27/Tensordot/GatherV2:output:0#dense_27/Tensordot/Const_2:output:0)dense_27/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_27/Tensordot/concat_1?
dense_27/TensordotReshape#dense_27/Tensordot/MatMul:product:0$dense_27/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????#2
dense_27/Tensordot?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/Tensordot:output:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????#2
dense_27/BiasAddx
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*,
_output_shapes
:??????????#2
dense_27/Relu?
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource*
_output_shapes

:#*
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_28/Tensordot/free
dense_28/Tensordot/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposedense_27/Relu:activations:0"dense_28/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????#2
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_28/BiasAddx
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_28/Relu?
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes?
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape?
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axis?
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2?
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis?
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const?
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod?
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1?
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1?
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axis?
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat?
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stack?
dense_29/Tensordot/transpose	Transposedense_28/Relu:activations:0"dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_29/Tensordot/transpose?
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_29/Tensordot/Reshape?
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_29/Tensordot/MatMul?
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
dense_29/Tensordot/Const_2?
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axis?
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1?
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
dense_29/Tensordot?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
dense_29/BiasAddx
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
dense_29/Relu?
lambda_4/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_4/Min/reduction_indices?
lambda_4/MinMindense_29/Relu:activations:0'lambda_4/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
lambda_4/Min?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMullambda_4/Min:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_30/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_31/BiasAdds
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_31/Relu?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMuldense_31/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAddt
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_32/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd}
dense_33/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_33/Sigmoid?
IdentityIdentitydense_33/Sigmoid:y:0 ^dense_27/BiasAdd/ReadVariableOp"^dense_27/Tensordot/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp"^dense_28/Tensordot/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp"^dense_29/Tensordot/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp0^fully_connected2_4/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2F
!dense_27/Tensordot/ReadVariableOp!dense_27/Tensordot/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/Tensordot/ReadVariableOp!dense_28/Tensordot/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2F
!dense_29/Tensordot/ReadVariableOp!dense_29/Tensordot/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2b
/fully_connected2_4/einsum/Einsum/ReadVariableOp/fully_connected2_4/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
C__inference_dense_29_layer_call_and_return_conditional_losses_34222

inputs3
!tensordot_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_lambda_4_layer_call_fn_34232

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_334592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
'__inference_model_4_layer_call_fn_33405
input_5
unknown:2
	unknown_0:2#
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:

	unknown_5:

	unknown_6:
2
	unknown_7:2
	unknown_8:2d
	unknown_9:d

unknown_10:	d?

unknown_11:	?

unknown_12:
??

unknown_13:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_333722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_5
?
_
C__inference_lambda_4_layer_call_and_return_conditional_losses_33459

inputs
identityp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Min/reduction_indicesk
MinMininputsMin/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
Min`
IdentityIdentityMin:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
D
(__inference_lambda_4_layer_call_fn_34227

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_4_layer_call_and_return_conditional_losses_333012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_34102

inputs7
%einsum_einsum_readvariableop_resource:2
identity??einsum/Einsum/ReadVariableOp?
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:2*
dtype02
einsum/Einsum/ReadVariableOp?
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????2*
equationijk,kl->ijl2
einsum/Einsum?
IdentityIdentityeinsum/Einsum:output:0^einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:??????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_30_layer_call_and_return_conditional_losses_34264

inputs0
matmul_readvariableop_resource:
2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_55
serving_default_input_5:0??????????=
dense_331
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?3
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?/
_tf_keras_network?/{"name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 500, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "FullyConnected2", "config": {"layer was saved without config": true}, "name": "fully_connected2_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_4", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_4", "inbound_nodes": [["fully_connected2_4", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["tf.nn.relu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMQAAAAdABqAWoCfABkAWQCjQJTACkDTukBAAAAKQHa\nBGF4aXMpA9oCdGbaBG1hdGjaCnJlZHVjZV9taW4pAdoBeKkAcgcAAAD6HzxpcHl0aG9uLWlucHV0\nLTIxLWUxMzAxN2U2NTg5Yj7aCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAADAAAAQwAAAHMQAAAAfABkARkAfABkAhkAZgJTACkDTukAAAAA6QIA\nAACpACkB2gVzaGFwZXIDAAAAcgMAAAD6HzxpcHl0aG9uLWlucHV0LTIxLWUxMzAxN2U2NTg5Yj7a\nCDxsYW1iZGE+GQAAAHMCAAAAAAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda_4", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["lambda_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["dense_31", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 2500, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["dense_33", 0, 0]]}, "shared_object_id": 24, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 500, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 500, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 500, 3]}, "float32", "input_5"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.007000000216066837, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 500, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 500, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?
	gamma
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "fully_connected2_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "FullyConnected2", "config": {"layer was saved without config": true}}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.nn.relu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_4", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "inbound_nodes": [["fully_connected2_4", 0, 0, {"name": null}]], "shared_object_id": 1}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.nn.relu_4", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500, 50]}}
?	

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_27", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500, 35]}}
?	

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_28", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500, 20]}}
?	
*trainable_variables
+	variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMQAAAAdABqAWoCfABkAWQCjQJTACkDTukBAAAAKQHa\nBGF4aXMpA9oCdGbaBG1hdGjaCnJlZHVjZV9taW4pAdoBeKkAcgcAAAD6HzxpcHl0aG9uLWlucHV0\nLTIxLWUxMzAxN2U2NTg5Yj7aCDxsYW1iZGE+GAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAADAAAAQwAAAHMQAAAAfABkARkAfABkAhkAZgJTACkDTukAAAAA6QIA\nAACpACkB2gVzaGFwZXIDAAAAcgMAAAD6HzxpcHl0aG9uLWlucHV0LTIxLWUxMzAxN2U2NTg5Yj7a\nCDxsYW1iZGE+GQAAAHMCAAAAAAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "inbound_nodes": [[["dense_29", 0, 0, {}]]], "shared_object_id": 11}
?	

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lambda_4", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?	

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_30", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?	

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_31", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?	

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 2500, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_32", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_rate
Kmomentum_cachem?m?m?m?m?$m?%m?.m?/m?4m?5m?:m?;m?@m?Am?v?v?v?v?v?$v?%v?.v?/v?4v?5v?:v?;v?@v?Av?"
	optimizer
?
0
1
2
3
4
$5
%6
.7
/8
49
510
:11
;12
@13
A14"
trackable_list_wrapper
?
0
1
2
3
4
$5
%6
.7
/8
49
510
:11
;12
@13
A14"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lnon_trainable_variables
trainable_variables
Mlayer_metrics
	variables

Nlayers
Olayer_regularization_losses
Pmetrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:22gamma
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qlayer_metrics
trainable_variables
Rlayer_regularization_losses
	variables

Slayers
Tnon_trainable_variables
Umetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
!:2#2dense_27/kernel
:#2dense_27/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vlayer_metrics
trainable_variables
Wlayer_regularization_losses
	variables

Xlayers
Ynon_trainable_variables
Zmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:#2dense_28/kernel
:2dense_28/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[layer_metrics
 trainable_variables
\layer_regularization_losses
!	variables

]layers
^non_trainable_variables
_metrics
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_29/kernel
:
2dense_29/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`layer_metrics
&trainable_variables
alayer_regularization_losses
'	variables

blayers
cnon_trainable_variables
dmetrics
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
elayer_metrics
*trainable_variables
flayer_regularization_losses
+	variables

glayers
hnon_trainable_variables
imetrics
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
22dense_30/kernel
:22dense_30/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jlayer_metrics
0trainable_variables
klayer_regularization_losses
1	variables

llayers
mnon_trainable_variables
nmetrics
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2d2dense_31/kernel
:d2dense_31/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
olayer_metrics
6trainable_variables
player_regularization_losses
7	variables

qlayers
rnon_trainable_variables
smetrics
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	d?2dense_32/kernel
:?2dense_32/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tlayer_metrics
<trainable_variables
ulayer_regularization_losses
=	variables

vlayers
wnon_trainable_variables
xmetrics
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_33/kernel
:?2dense_33/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ylayer_metrics
Btrainable_variables
zlayer_regularization_losses
C	variables

{layers
|non_trainable_variables
}metrics
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
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
10"
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
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
?
	total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 33}
:  (2total
:  (2count
/
0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:22Nadam/gamma/m
':%2#2Nadam/dense_27/kernel/m
!:#2Nadam/dense_27/bias/m
':%#2Nadam/dense_28/kernel/m
!:2Nadam/dense_28/bias/m
':%
2Nadam/dense_29/kernel/m
!:
2Nadam/dense_29/bias/m
':%
22Nadam/dense_30/kernel/m
!:22Nadam/dense_30/bias/m
':%2d2Nadam/dense_31/kernel/m
!:d2Nadam/dense_31/bias/m
(:&	d?2Nadam/dense_32/kernel/m
": ?2Nadam/dense_32/bias/m
):'
??2Nadam/dense_33/kernel/m
": ?2Nadam/dense_33/bias/m
:22Nadam/gamma/v
':%2#2Nadam/dense_27/kernel/v
!:#2Nadam/dense_27/bias/v
':%#2Nadam/dense_28/kernel/v
!:2Nadam/dense_28/bias/v
':%
2Nadam/dense_29/kernel/v
!:
2Nadam/dense_29/bias/v
':%
22Nadam/dense_30/kernel/v
!:22Nadam/dense_30/bias/v
':%2d2Nadam/dense_31/kernel/v
!:d2Nadam/dense_31/bias/v
(:&	d?2Nadam/dense_32/kernel/v
": ?2Nadam/dense_32/bias/v
):'
??2Nadam/dense_33/kernel/v
": ?2Nadam/dense_33/bias/v
?2?
'__inference_model_4_layer_call_fn_33405
'__inference_model_4_layer_call_fn_33815
'__inference_model_4_layer_call_fn_33850
'__inference_model_4_layer_call_fn_33649?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_4_layer_call_and_return_conditional_losses_33969
B__inference_model_4_layer_call_and_return_conditional_losses_34088
B__inference_model_4_layer_call_and_return_conditional_losses_33693
B__inference_model_4_layer_call_and_return_conditional_losses_33737?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_33165?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_5??????????
?2?
2__inference_fully_connected2_4_layer_call_fn_34095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_34102?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_27_layer_call_fn_34111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_27_layer_call_and_return_conditional_losses_34142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_28_layer_call_fn_34151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_28_layer_call_and_return_conditional_losses_34182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_29_layer_call_fn_34191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_29_layer_call_and_return_conditional_losses_34222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_lambda_4_layer_call_fn_34227
(__inference_lambda_4_layer_call_fn_34232?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lambda_4_layer_call_and_return_conditional_losses_34238
C__inference_lambda_4_layer_call_and_return_conditional_losses_34244?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_30_layer_call_fn_34253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_30_layer_call_and_return_conditional_losses_34264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_31_layer_call_fn_34273?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_31_layer_call_and_return_conditional_losses_34284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_32_layer_call_fn_34293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_32_layer_call_and_return_conditional_losses_34304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_33_layer_call_fn_34313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_33_layer_call_and_return_conditional_losses_34324?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_33780input_5"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_33165~$%./45:;@A5?2
+?(
&?#
input_5??????????
? "4?1
/
dense_33#? 
dense_33???????????
C__inference_dense_27_layer_call_and_return_conditional_losses_34142f4?1
*?'
%?"
inputs??????????2
? "*?'
 ?
0??????????#
? ?
(__inference_dense_27_layer_call_fn_34111Y4?1
*?'
%?"
inputs??????????2
? "???????????#?
C__inference_dense_28_layer_call_and_return_conditional_losses_34182f4?1
*?'
%?"
inputs??????????#
? "*?'
 ?
0??????????
? ?
(__inference_dense_28_layer_call_fn_34151Y4?1
*?'
%?"
inputs??????????#
? "????????????
C__inference_dense_29_layer_call_and_return_conditional_losses_34222f$%4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????

? ?
(__inference_dense_29_layer_call_fn_34191Y$%4?1
*?'
%?"
inputs??????????
? "???????????
?
C__inference_dense_30_layer_call_and_return_conditional_losses_34264\.//?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????2
? {
(__inference_dense_30_layer_call_fn_34253O.//?,
%?"
 ?
inputs?????????

? "??????????2?
C__inference_dense_31_layer_call_and_return_conditional_losses_34284\45/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????d
? {
(__inference_dense_31_layer_call_fn_34273O45/?,
%?"
 ?
inputs?????????2
? "??????????d?
C__inference_dense_32_layer_call_and_return_conditional_losses_34304]:;/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? |
(__inference_dense_32_layer_call_fn_34293P:;/?,
%?"
 ?
inputs?????????d
? "????????????
C__inference_dense_33_layer_call_and_return_conditional_losses_34324^@A0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_33_layer_call_fn_34313Q@A0?-
&?#
!?
inputs??????????
? "????????????
M__inference_fully_connected2_4_layer_call_and_return_conditional_losses_34102e4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????2
? ?
2__inference_fully_connected2_4_layer_call_fn_34095X4?1
*?'
%?"
inputs??????????
? "???????????2?
C__inference_lambda_4_layer_call_and_return_conditional_losses_34238e<?9
2?/
%?"
inputs??????????


 
p 
? "%?"
?
0?????????

? ?
C__inference_lambda_4_layer_call_and_return_conditional_losses_34244e<?9
2?/
%?"
inputs??????????


 
p
? "%?"
?
0?????????

? ?
(__inference_lambda_4_layer_call_fn_34227X<?9
2?/
%?"
inputs??????????


 
p 
? "??????????
?
(__inference_lambda_4_layer_call_fn_34232X<?9
2?/
%?"
inputs??????????


 
p
? "??????????
?
B__inference_model_4_layer_call_and_return_conditional_losses_33693x$%./45:;@A=?:
3?0
&?#
input_5??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_33737x$%./45:;@A=?:
3?0
&?#
input_5??????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_33969w$%./45:;@A<?9
2?/
%?"
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_34088w$%./45:;@A<?9
2?/
%?"
inputs??????????
p

 
? "&?#
?
0??????????
? ?
'__inference_model_4_layer_call_fn_33405k$%./45:;@A=?:
3?0
&?#
input_5??????????
p 

 
? "????????????
'__inference_model_4_layer_call_fn_33649k$%./45:;@A=?:
3?0
&?#
input_5??????????
p

 
? "????????????
'__inference_model_4_layer_call_fn_33815j$%./45:;@A<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
'__inference_model_4_layer_call_fn_33850j$%./45:;@A<?9
2?/
%?"
inputs??????????
p

 
? "????????????
#__inference_signature_wrapper_33780?$%./45:;@A@?=
? 
6?3
1
input_5&?#
input_5??????????"4?1
/
dense_33#? 
dense_33??????????