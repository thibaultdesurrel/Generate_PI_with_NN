??
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
:*
shared_namegamma
_
gamma/Read/ReadVariableOpReadVariableOpgamma*
_output_shapes

:*
dtype0
z
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_49/kernel
s
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel*
_output_shapes

:*
dtype0
r
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_49/bias
k
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
_output_shapes
:*
dtype0
z
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_50/kernel
s
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes

:
*
dtype0
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
:
*
dtype0
z
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2* 
shared_namedense_51/kernel
s
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes

:
2*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:2*
dtype0
z
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d* 
shared_namedense_52/kernel
s
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes

:2d*
dtype0
r
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_52/bias
k
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes
:d*
dtype0
{
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?* 
shared_namedense_53/kernel
t
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes
:	d?*
dtype0
s
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_53/bias
l
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes	
:?*
dtype0
|
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_54/kernel
u
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel* 
_output_shapes
:
??*
dtype0
s
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_54/bias
l
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
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
:*
shared_nameNadam/gamma/m
o
!Nadam/gamma/m/Read/ReadVariableOpReadVariableOpNadam/gamma/m*
_output_shapes

:*
dtype0
?
Nadam/dense_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameNadam/dense_49/kernel/m
?
+Nadam/dense_49/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_49/kernel/m*
_output_shapes

:*
dtype0
?
Nadam/dense_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_49/bias/m
{
)Nadam/dense_49/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_49/bias/m*
_output_shapes
:*
dtype0
?
Nadam/dense_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_50/kernel/m
?
+Nadam/dense_50/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_50/kernel/m*
_output_shapes

:
*
dtype0
?
Nadam/dense_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_50/bias/m
{
)Nadam/dense_50/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_50/bias/m*
_output_shapes
:
*
dtype0
?
Nadam/dense_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*(
shared_nameNadam/dense_51/kernel/m
?
+Nadam/dense_51/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_51/kernel/m*
_output_shapes

:
2*
dtype0
?
Nadam/dense_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameNadam/dense_51/bias/m
{
)Nadam/dense_51/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_51/bias/m*
_output_shapes
:2*
dtype0
?
Nadam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*(
shared_nameNadam/dense_52/kernel/m
?
+Nadam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_52/kernel/m*
_output_shapes

:2d*
dtype0
?
Nadam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_52/bias/m
{
)Nadam/dense_52/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_52/bias/m*
_output_shapes
:d*
dtype0
?
Nadam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameNadam/dense_53/kernel/m
?
+Nadam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_53/kernel/m*
_output_shapes
:	d?*
dtype0
?
Nadam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_53/bias/m
|
)Nadam/dense_53/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_53/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_54/kernel/m
?
+Nadam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_54/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_54/bias/m
|
)Nadam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_54/bias/m*
_output_shapes	
:?*
dtype0
v
Nadam/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameNadam/gamma/v
o
!Nadam/gamma/v/Read/ReadVariableOpReadVariableOpNadam/gamma/v*
_output_shapes

:*
dtype0
?
Nadam/dense_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameNadam/dense_49/kernel/v
?
+Nadam/dense_49/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_49/kernel/v*
_output_shapes

:*
dtype0
?
Nadam/dense_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_49/bias/v
{
)Nadam/dense_49/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_49/bias/v*
_output_shapes
:*
dtype0
?
Nadam/dense_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_50/kernel/v
?
+Nadam/dense_50/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_50/kernel/v*
_output_shapes

:
*
dtype0
?
Nadam/dense_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_50/bias/v
{
)Nadam/dense_50/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_50/bias/v*
_output_shapes
:
*
dtype0
?
Nadam/dense_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*(
shared_nameNadam/dense_51/kernel/v
?
+Nadam/dense_51/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_51/kernel/v*
_output_shapes

:
2*
dtype0
?
Nadam/dense_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameNadam/dense_51/bias/v
{
)Nadam/dense_51/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_51/bias/v*
_output_shapes
:2*
dtype0
?
Nadam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*(
shared_nameNadam/dense_52/kernel/v
?
+Nadam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_52/kernel/v*
_output_shapes

:2d*
dtype0
?
Nadam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameNadam/dense_52/bias/v
{
)Nadam/dense_52/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_52/bias/v*
_output_shapes
:d*
dtype0
?
Nadam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_nameNadam/dense_53/kernel/v
?
+Nadam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_53/kernel/v*
_output_shapes
:	d?*
dtype0
?
Nadam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_53/bias/v
|
)Nadam/dense_53/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_53/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_54/kernel/v
?
+Nadam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_54/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_54/bias/v
|
)Nadam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_54/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?F
value?FB?F B?F
?
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
	layer_with_weights-5
	layer-8

layer_with_weights-6
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
]
	gamma
trainable_variables
	variables
regularization_losses
	keras_api

	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate
Dmomentum_cachemwmxmymzm{'m|(m}-m~.m3m?4m?9m?:m?v?v?v?v?v?'v?(v?-v?.v?3v?4v?9v?:v?
^
0
1
2
3
4
'5
(6
-7
.8
39
410
911
:12
^
0
1
2
3
4
'5
(6
-7
.8
39
410
911
:12
 
?
trainable_variables
	variables
Elayer_metrics
Fmetrics

Glayers
Hnon_trainable_variables
Ilayer_regularization_losses
regularization_losses
 
PN
VARIABLE_VALUEgamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
trainable_variables
	variables
Jlayer_metrics
Kmetrics

Llayers
Mnon_trainable_variables
Nlayer_regularization_losses
regularization_losses
 
[Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_49/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
Olayer_metrics
Pmetrics

Qlayers
Rnon_trainable_variables
Slayer_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
 	variables
Tlayer_metrics
Umetrics

Vlayers
Wnon_trainable_variables
Xlayer_regularization_losses
!regularization_losses
 
 
 
?
#trainable_variables
$	variables
Ylayer_metrics
Zmetrics

[layers
\non_trainable_variables
]layer_regularization_losses
%regularization_losses
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)trainable_variables
*	variables
^layer_metrics
_metrics

`layers
anon_trainable_variables
blayer_regularization_losses
+regularization_losses
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
/trainable_variables
0	variables
clayer_metrics
dmetrics

elayers
fnon_trainable_variables
glayer_regularization_losses
1regularization_losses
[Y
VARIABLE_VALUEdense_53/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_53/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
5trainable_variables
6	variables
hlayer_metrics
imetrics

jlayers
knon_trainable_variables
llayer_regularization_losses
7regularization_losses
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
;trainable_variables
<	variables
mlayer_metrics
nmetrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
=regularization_losses
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

r0
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
4
	stotal
	tcount
u	variables
v	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

u	variables
tr
VARIABLE_VALUENadam/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_49/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_49/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_50/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_50/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_51/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_51/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_52/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_52/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_53/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_53/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_54/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_54/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUENadam/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_49/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_49/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_50/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_50/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_51/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_51/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_52/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_52/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_53/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_53/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_54/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_54/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_9Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9gammadense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_41561
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamegamma/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp!Nadam/gamma/m/Read/ReadVariableOp+Nadam/dense_49/kernel/m/Read/ReadVariableOp)Nadam/dense_49/bias/m/Read/ReadVariableOp+Nadam/dense_50/kernel/m/Read/ReadVariableOp)Nadam/dense_50/bias/m/Read/ReadVariableOp+Nadam/dense_51/kernel/m/Read/ReadVariableOp)Nadam/dense_51/bias/m/Read/ReadVariableOp+Nadam/dense_52/kernel/m/Read/ReadVariableOp)Nadam/dense_52/bias/m/Read/ReadVariableOp+Nadam/dense_53/kernel/m/Read/ReadVariableOp)Nadam/dense_53/bias/m/Read/ReadVariableOp+Nadam/dense_54/kernel/m/Read/ReadVariableOp)Nadam/dense_54/bias/m/Read/ReadVariableOp!Nadam/gamma/v/Read/ReadVariableOp+Nadam/dense_49/kernel/v/Read/ReadVariableOp)Nadam/dense_49/bias/v/Read/ReadVariableOp+Nadam/dense_50/kernel/v/Read/ReadVariableOp)Nadam/dense_50/bias/v/Read/ReadVariableOp+Nadam/dense_51/kernel/v/Read/ReadVariableOp)Nadam/dense_51/bias/v/Read/ReadVariableOp+Nadam/dense_52/kernel/v/Read/ReadVariableOp)Nadam/dense_52/bias/v/Read/ReadVariableOp+Nadam/dense_53/kernel/v/Read/ReadVariableOp)Nadam/dense_53/bias/v/Read/ReadVariableOp+Nadam/dense_54/kernel/v/Read/ReadVariableOp)Nadam/dense_54/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
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
__inference__traced_save_42167
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegammadense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/gamma/mNadam/dense_49/kernel/mNadam/dense_49/bias/mNadam/dense_50/kernel/mNadam/dense_50/bias/mNadam/dense_51/kernel/mNadam/dense_51/bias/mNadam/dense_52/kernel/mNadam/dense_52/bias/mNadam/dense_53/kernel/mNadam/dense_53/bias/mNadam/dense_54/kernel/mNadam/dense_54/bias/mNadam/gamma/vNadam/dense_49/kernel/vNadam/dense_49/bias/vNadam/dense_50/kernel/vNadam/dense_50/bias/vNadam/dense_51/kernel/vNadam/dense_51/bias/vNadam/dense_52/kernel/vNadam/dense_52/bias/vNadam/dense_53/kernel/vNadam/dense_53/bias/vNadam/dense_54/kernel/vNadam/dense_54/bias/v*;
Tin4
220*
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
!__inference__traced_restore_42318??	
?z
?

B__inference_model_8_layer_call_and_return_conditional_losses_41745

inputsJ
8fully_connected2_8_einsum_einsum_readvariableop_resource:<
*dense_49_tensordot_readvariableop_resource:6
(dense_49_biasadd_readvariableop_resource:<
*dense_50_tensordot_readvariableop_resource:
6
(dense_50_biasadd_readvariableop_resource:
9
'dense_51_matmul_readvariableop_resource:
26
(dense_51_biasadd_readvariableop_resource:29
'dense_52_matmul_readvariableop_resource:2d6
(dense_52_biasadd_readvariableop_resource:d:
'dense_53_matmul_readvariableop_resource:	d?7
(dense_53_biasadd_readvariableop_resource:	?;
'dense_54_matmul_readvariableop_resource:
??7
(dense_54_biasadd_readvariableop_resource:	?
identity??dense_49/BiasAdd/ReadVariableOp?!dense_49/Tensordot/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?!dense_50/Tensordot/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?/fully_connected2_8/einsum/Einsum/ReadVariableOp?
/fully_connected2_8/einsum/Einsum/ReadVariableOpReadVariableOp8fully_connected2_8_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype021
/fully_connected2_8/einsum/Einsum/ReadVariableOp?
 fully_connected2_8/einsum/EinsumEinsuminputs7fully_connected2_8/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????*
equationijk,kl->ijl2"
 fully_connected2_8/einsum/Einsum?
tf.nn.relu_8/ReluRelu)fully_connected2_8/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
!dense_49/Tensordot/ReadVariableOpReadVariableOp*dense_49_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_49/Tensordot/ReadVariableOp|
dense_49/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_49/Tensordot/axes?
dense_49/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_49/Tensordot/free?
dense_49/Tensordot/ShapeShapetf.nn.relu_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_49/Tensordot/Shape?
 dense_49/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_49/Tensordot/GatherV2/axis?
dense_49/Tensordot/GatherV2GatherV2!dense_49/Tensordot/Shape:output:0 dense_49/Tensordot/free:output:0)dense_49/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_49/Tensordot/GatherV2?
"dense_49/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_49/Tensordot/GatherV2_1/axis?
dense_49/Tensordot/GatherV2_1GatherV2!dense_49/Tensordot/Shape:output:0 dense_49/Tensordot/axes:output:0+dense_49/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_49/Tensordot/GatherV2_1~
dense_49/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_49/Tensordot/Const?
dense_49/Tensordot/ProdProd$dense_49/Tensordot/GatherV2:output:0!dense_49/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_49/Tensordot/Prod?
dense_49/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_49/Tensordot/Const_1?
dense_49/Tensordot/Prod_1Prod&dense_49/Tensordot/GatherV2_1:output:0#dense_49/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_49/Tensordot/Prod_1?
dense_49/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_49/Tensordot/concat/axis?
dense_49/Tensordot/concatConcatV2 dense_49/Tensordot/free:output:0 dense_49/Tensordot/axes:output:0'dense_49/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/concat?
dense_49/Tensordot/stackPack dense_49/Tensordot/Prod:output:0"dense_49/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/stack?
dense_49/Tensordot/transpose	Transposetf.nn.relu_8/Relu:activations:0"dense_49/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Tensordot/transpose?
dense_49/Tensordot/ReshapeReshape dense_49/Tensordot/transpose:y:0!dense_49/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_49/Tensordot/Reshape?
dense_49/Tensordot/MatMulMatMul#dense_49/Tensordot/Reshape:output:0)dense_49/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_49/Tensordot/MatMul?
dense_49/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_49/Tensordot/Const_2?
 dense_49/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_49/Tensordot/concat_1/axis?
dense_49/Tensordot/concat_1ConcatV2$dense_49/Tensordot/GatherV2:output:0#dense_49/Tensordot/Const_2:output:0)dense_49/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/concat_1?
dense_49/TensordotReshape#dense_49/Tensordot/MatMul:product:0$dense_49/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Tensordot?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/Tensordot:output:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_49/BiasAddx
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Relu?
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_50/Tensordot/axes?
dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_50/Tensordot/free
dense_50/Tensordot/ShapeShapedense_49/Relu:activations:0*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape?
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axis?
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2?
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis?
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const?
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod?
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1?
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1?
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axis?
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat?
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stack?
dense_50/Tensordot/transpose	Transposedense_49/Relu:activations:0"dense_50/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_50/Tensordot/transpose?
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_50/Tensordot/Reshape?
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_50/Tensordot/MatMul?
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
dense_50/Tensordot/Const_2?
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axis?
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1?
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
dense_50/Tensordot?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
dense_50/BiasAddx
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
dense_50/Relu?
lambda_8/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_8/Min/reduction_indices?
lambda_8/MinMindense_50/Relu:activations:0'lambda_8/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
lambda_8/Min?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMullambda_8/Min:output:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_51/BiasAdds
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_51/Relu?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMuldense_51/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_52/BiasAdds
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_52/Relu?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_53/MatMul/ReadVariableOp?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_53/MatMul?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_53/BiasAddt
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_53/Relu?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMuldense_53/Relu:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_54/BiasAdd}
dense_54/SigmoidSigmoiddense_54/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_54/Sigmoid?
IdentityIdentitydense_54/Sigmoid:y:0 ^dense_49/BiasAdd/ReadVariableOp"^dense_49/Tensordot/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp"^dense_50/Tensordot/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp0^fully_connected2_8/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2F
!dense_49/Tensordot/ReadVariableOp!dense_49/Tensordot/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2F
!dense_50/Tensordot/ReadVariableOp!dense_50/Tensordot/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2b
/fully_connected2_8/einsum/Einsum/ReadVariableOp/fully_connected2_8/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
B__inference_model_8_layer_call_and_return_conditional_losses_41384

inputs*
fully_connected2_8_41348: 
dense_49_41352:
dense_49_41354: 
dense_50_41357:

dense_50_41359:
 
dense_51_41363:
2
dense_51_41365:2 
dense_52_41368:2d
dense_52_41370:d!
dense_53_41373:	d?
dense_53_41375:	?"
dense_54_41378:
??
dense_54_41380:	?
identity?? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?*fully_connected2_8/StatefulPartitionedCall?
*fully_connected2_8/StatefulPartitionedCallStatefulPartitionedCallinputsfully_connected2_8_41348*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_410422,
*fully_connected2_8/StatefulPartitionedCall?
tf.nn.relu_8/ReluRelu3fully_connected2_8/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
 dense_49/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0dense_49_41352dense_49_41354*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_49_layer_call_and_return_conditional_losses_410782"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_41357dense_50_41359*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_411152"
 dense_50/StatefulPartitionedCall?
lambda_8/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
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
C__inference_lambda_8_layer_call_and_return_conditional_losses_412812
lambda_8/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0dense_51_41363dense_51_41365*
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
C__inference_dense_51_layer_call_and_return_conditional_losses_411402"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_41368dense_52_41370*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_411572"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_41373dense_53_41375*
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
C__inference_dense_53_layer_call_and_return_conditional_losses_411742"
 dense_53/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_41378dense_54_41380*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_411912"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall+^fully_connected2_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2X
*fully_connected2_8/StatefulPartitionedCall*fully_connected2_8/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
B__inference_model_8_layer_call_and_return_conditional_losses_41198

inputs*
fully_connected2_8_41043: 
dense_49_41079:
dense_49_41081: 
dense_50_41116:

dense_50_41118:
 
dense_51_41141:
2
dense_51_41143:2 
dense_52_41158:2d
dense_52_41160:d!
dense_53_41175:	d?
dense_53_41177:	?"
dense_54_41192:
??
dense_54_41194:	?
identity?? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?*fully_connected2_8/StatefulPartitionedCall?
*fully_connected2_8/StatefulPartitionedCallStatefulPartitionedCallinputsfully_connected2_8_41043*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_410422,
*fully_connected2_8/StatefulPartitionedCall?
tf.nn.relu_8/ReluRelu3fully_connected2_8/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
 dense_49/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0dense_49_41079dense_49_41081*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_49_layer_call_and_return_conditional_losses_410782"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_41116dense_50_41118*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_411152"
 dense_50/StatefulPartitionedCall?
lambda_8/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
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
C__inference_lambda_8_layer_call_and_return_conditional_losses_411272
lambda_8/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0dense_51_41141dense_51_41143*
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
C__inference_dense_51_layer_call_and_return_conditional_losses_411402"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_41158dense_52_41160*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_411572"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_41175dense_53_41177*
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
C__inference_dense_53_layer_call_and_return_conditional_losses_411742"
 dense_53/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_41192dense_54_41194*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_411912"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall+^fully_connected2_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2X
*fully_connected2_8/StatefulPartitionedCall*fully_connected2_8/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_model_8_layer_call_fn_41776

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:

	unknown_3:

	unknown_4:
2
	unknown_5:2
	unknown_6:2d
	unknown_7:d
	unknown_8:	d?
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_411982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_52_layer_call_and_return_conditional_losses_41157

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
(__inference_dense_51_layer_call_fn_41943

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
C__inference_dense_51_layer_call_and_return_conditional_losses_411402
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
?+
?
B__inference_model_8_layer_call_and_return_conditional_losses_41483
input_9*
fully_connected2_8_41447: 
dense_49_41451:
dense_49_41453: 
dense_50_41456:

dense_50_41458:
 
dense_51_41462:
2
dense_51_41464:2 
dense_52_41467:2d
dense_52_41469:d!
dense_53_41472:	d?
dense_53_41474:	?"
dense_54_41477:
??
dense_54_41479:	?
identity?? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?*fully_connected2_8/StatefulPartitionedCall?
*fully_connected2_8/StatefulPartitionedCallStatefulPartitionedCallinput_9fully_connected2_8_41447*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_410422,
*fully_connected2_8/StatefulPartitionedCall?
tf.nn.relu_8/ReluRelu3fully_connected2_8/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
 dense_49/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0dense_49_41451dense_49_41453*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_49_layer_call_and_return_conditional_losses_410782"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_41456dense_50_41458*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_411152"
 dense_50/StatefulPartitionedCall?
lambda_8/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
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
C__inference_lambda_8_layer_call_and_return_conditional_losses_411272
lambda_8/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0dense_51_41462dense_51_41464*
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
C__inference_dense_51_layer_call_and_return_conditional_losses_411402"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_41467dense_52_41469*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_411572"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_41472dense_53_41474*
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
C__inference_dense_53_layer_call_and_return_conditional_losses_411742"
 dense_53/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_41477dense_54_41479*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_411912"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall+^fully_connected2_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2X
*fully_connected2_8/StatefulPartitionedCall*fully_connected2_8/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
?+
?
B__inference_model_8_layer_call_and_return_conditional_losses_41522
input_9*
fully_connected2_8_41486: 
dense_49_41490:
dense_49_41492: 
dense_50_41495:

dense_50_41497:
 
dense_51_41501:
2
dense_51_41503:2 
dense_52_41506:2d
dense_52_41508:d!
dense_53_41511:	d?
dense_53_41513:	?"
dense_54_41516:
??
dense_54_41518:	?
identity?? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?*fully_connected2_8/StatefulPartitionedCall?
*fully_connected2_8/StatefulPartitionedCallStatefulPartitionedCallinput_9fully_connected2_8_41486*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_410422,
*fully_connected2_8/StatefulPartitionedCall?
tf.nn.relu_8/ReluRelu3fully_connected2_8/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
 dense_49/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0dense_49_41490dense_49_41492*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_49_layer_call_and_return_conditional_losses_410782"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_41495dense_50_41497*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_411152"
 dense_50/StatefulPartitionedCall?
lambda_8/PartitionedCallPartitionedCall)dense_50/StatefulPartitionedCall:output:0*
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
C__inference_lambda_8_layer_call_and_return_conditional_losses_412812
lambda_8/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0dense_51_41501dense_51_41503*
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
C__inference_dense_51_layer_call_and_return_conditional_losses_411402"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_41506dense_52_41508*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_411572"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_41511dense_53_41513*
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
C__inference_dense_53_layer_call_and_return_conditional_losses_411742"
 dense_53/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_41516dense_54_41518*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_411912"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall+^fully_connected2_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2X
*fully_connected2_8/StatefulPartitionedCall*fully_connected2_8/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
?z
?

B__inference_model_8_layer_call_and_return_conditional_losses_41653

inputsJ
8fully_connected2_8_einsum_einsum_readvariableop_resource:<
*dense_49_tensordot_readvariableop_resource:6
(dense_49_biasadd_readvariableop_resource:<
*dense_50_tensordot_readvariableop_resource:
6
(dense_50_biasadd_readvariableop_resource:
9
'dense_51_matmul_readvariableop_resource:
26
(dense_51_biasadd_readvariableop_resource:29
'dense_52_matmul_readvariableop_resource:2d6
(dense_52_biasadd_readvariableop_resource:d:
'dense_53_matmul_readvariableop_resource:	d?7
(dense_53_biasadd_readvariableop_resource:	?;
'dense_54_matmul_readvariableop_resource:
??7
(dense_54_biasadd_readvariableop_resource:	?
identity??dense_49/BiasAdd/ReadVariableOp?!dense_49/Tensordot/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?!dense_50/Tensordot/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?/fully_connected2_8/einsum/Einsum/ReadVariableOp?
/fully_connected2_8/einsum/Einsum/ReadVariableOpReadVariableOp8fully_connected2_8_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype021
/fully_connected2_8/einsum/Einsum/ReadVariableOp?
 fully_connected2_8/einsum/EinsumEinsuminputs7fully_connected2_8/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????*
equationijk,kl->ijl2"
 fully_connected2_8/einsum/Einsum?
tf.nn.relu_8/ReluRelu)fully_connected2_8/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
!dense_49/Tensordot/ReadVariableOpReadVariableOp*dense_49_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_49/Tensordot/ReadVariableOp|
dense_49/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_49/Tensordot/axes?
dense_49/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_49/Tensordot/free?
dense_49/Tensordot/ShapeShapetf.nn.relu_8/Relu:activations:0*
T0*
_output_shapes
:2
dense_49/Tensordot/Shape?
 dense_49/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_49/Tensordot/GatherV2/axis?
dense_49/Tensordot/GatherV2GatherV2!dense_49/Tensordot/Shape:output:0 dense_49/Tensordot/free:output:0)dense_49/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_49/Tensordot/GatherV2?
"dense_49/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_49/Tensordot/GatherV2_1/axis?
dense_49/Tensordot/GatherV2_1GatherV2!dense_49/Tensordot/Shape:output:0 dense_49/Tensordot/axes:output:0+dense_49/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_49/Tensordot/GatherV2_1~
dense_49/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_49/Tensordot/Const?
dense_49/Tensordot/ProdProd$dense_49/Tensordot/GatherV2:output:0!dense_49/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_49/Tensordot/Prod?
dense_49/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_49/Tensordot/Const_1?
dense_49/Tensordot/Prod_1Prod&dense_49/Tensordot/GatherV2_1:output:0#dense_49/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_49/Tensordot/Prod_1?
dense_49/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_49/Tensordot/concat/axis?
dense_49/Tensordot/concatConcatV2 dense_49/Tensordot/free:output:0 dense_49/Tensordot/axes:output:0'dense_49/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/concat?
dense_49/Tensordot/stackPack dense_49/Tensordot/Prod:output:0"dense_49/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/stack?
dense_49/Tensordot/transpose	Transposetf.nn.relu_8/Relu:activations:0"dense_49/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Tensordot/transpose?
dense_49/Tensordot/ReshapeReshape dense_49/Tensordot/transpose:y:0!dense_49/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_49/Tensordot/Reshape?
dense_49/Tensordot/MatMulMatMul#dense_49/Tensordot/Reshape:output:0)dense_49/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_49/Tensordot/MatMul?
dense_49/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_49/Tensordot/Const_2?
 dense_49/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_49/Tensordot/concat_1/axis?
dense_49/Tensordot/concat_1ConcatV2$dense_49/Tensordot/GatherV2:output:0#dense_49/Tensordot/Const_2:output:0)dense_49/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_49/Tensordot/concat_1?
dense_49/TensordotReshape#dense_49/Tensordot/MatMul:product:0$dense_49/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Tensordot?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/Tensordot:output:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_49/BiasAddx
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_49/Relu?
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_50/Tensordot/axes?
dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_50/Tensordot/free
dense_50/Tensordot/ShapeShapedense_49/Relu:activations:0*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape?
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axis?
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2?
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis?
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const?
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod?
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1?
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1?
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axis?
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat?
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stack?
dense_50/Tensordot/transpose	Transposedense_49/Relu:activations:0"dense_50/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_50/Tensordot/transpose?
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_50/Tensordot/Reshape?
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_50/Tensordot/MatMul?
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2
dense_50/Tensordot/Const_2?
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axis?
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1?
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
dense_50/Tensordot?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
dense_50/BiasAddx
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
dense_50/Relu?
lambda_8/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_8/Min/reduction_indices?
lambda_8/MinMindense_50/Relu:activations:0'lambda_8/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
lambda_8/Min?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMullambda_8/Min:output:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_51/BiasAdds
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
dense_51/Relu?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMuldense_51/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_52/BiasAdds
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_52/Relu?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02 
dense_53/MatMul/ReadVariableOp?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_53/MatMul?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_53/BiasAddt
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_53/Relu?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMuldense_53/Relu:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_54/BiasAdd}
dense_54/SigmoidSigmoiddense_54/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_54/Sigmoid?
IdentityIdentitydense_54/Sigmoid:y:0 ^dense_49/BiasAdd/ReadVariableOp"^dense_49/Tensordot/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp"^dense_50/Tensordot/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp0^fully_connected2_8/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2F
!dense_49/Tensordot/ReadVariableOp!dense_49/Tensordot/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2F
!dense_50/Tensordot/ReadVariableOp!dense_50/Tensordot/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2b
/fully_connected2_8/einsum/Einsum/ReadVariableOp/fully_connected2_8/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_model_8_layer_call_fn_41227
input_9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:

	unknown_3:

	unknown_4:
2
	unknown_5:2
	unknown_6:2d
	unknown_7:d
	unknown_8:	d?
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_411982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
?

?
'__inference_model_8_layer_call_fn_41444
input_9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:

	unknown_3:

	unknown_4:
2
	unknown_5:2
	unknown_6:2d
	unknown_7:d
	unknown_8:	d?
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_413842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
?
_
C__inference_lambda_8_layer_call_and_return_conditional_losses_41913

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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
_
C__inference_lambda_8_layer_call_and_return_conditional_losses_41907

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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
D
(__inference_lambda_8_layer_call_fn_41918

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
C__inference_lambda_8_layer_call_and_return_conditional_losses_411272
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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_41042

inputs7
%einsum_einsum_readvariableop_resource:
identity??einsum/Einsum/ReadVariableOp?
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp?
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????*
equationijk,kl->ijl2
einsum/Einsum?
IdentityIdentityeinsum/Einsum:output:0^einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_49_layer_call_fn_41861

inputs
unknown:
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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_49_layer_call_and_return_conditional_losses_410782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_lambda_8_layer_call_fn_41923

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
C__inference_lambda_8_layer_call_and_return_conditional_losses_412812
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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
2__inference_fully_connected2_8_layer_call_fn_41821

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_410422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
C__inference_dense_49_layer_call_and_return_conditional_losses_41852

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_41994

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
C__inference_dense_51_layer_call_and_return_conditional_losses_41934

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
??
?
!__inference__traced_restore_42318
file_prefix(
assignvariableop_gamma:4
"assignvariableop_1_dense_49_kernel:.
 assignvariableop_2_dense_49_bias:4
"assignvariableop_3_dense_50_kernel:
.
 assignvariableop_4_dense_50_bias:
4
"assignvariableop_5_dense_51_kernel:
2.
 assignvariableop_6_dense_51_bias:24
"assignvariableop_7_dense_52_kernel:2d.
 assignvariableop_8_dense_52_bias:d5
"assignvariableop_9_dense_53_kernel:	d?0
!assignvariableop_10_dense_53_bias:	?7
#assignvariableop_11_dense_54_kernel:
??0
!assignvariableop_12_dense_54_bias:	?(
assignvariableop_13_nadam_iter:	 *
 assignvariableop_14_nadam_beta_1: *
 assignvariableop_15_nadam_beta_2: )
assignvariableop_16_nadam_decay: 1
'assignvariableop_17_nadam_learning_rate: 2
(assignvariableop_18_nadam_momentum_cache: #
assignvariableop_19_total: #
assignvariableop_20_count: 3
!assignvariableop_21_nadam_gamma_m:=
+assignvariableop_22_nadam_dense_49_kernel_m:7
)assignvariableop_23_nadam_dense_49_bias_m:=
+assignvariableop_24_nadam_dense_50_kernel_m:
7
)assignvariableop_25_nadam_dense_50_bias_m:
=
+assignvariableop_26_nadam_dense_51_kernel_m:
27
)assignvariableop_27_nadam_dense_51_bias_m:2=
+assignvariableop_28_nadam_dense_52_kernel_m:2d7
)assignvariableop_29_nadam_dense_52_bias_m:d>
+assignvariableop_30_nadam_dense_53_kernel_m:	d?8
)assignvariableop_31_nadam_dense_53_bias_m:	??
+assignvariableop_32_nadam_dense_54_kernel_m:
??8
)assignvariableop_33_nadam_dense_54_bias_m:	?3
!assignvariableop_34_nadam_gamma_v:=
+assignvariableop_35_nadam_dense_49_kernel_v:7
)assignvariableop_36_nadam_dense_49_bias_v:=
+assignvariableop_37_nadam_dense_50_kernel_v:
7
)assignvariableop_38_nadam_dense_50_bias_v:
=
+assignvariableop_39_nadam_dense_51_kernel_v:
27
)assignvariableop_40_nadam_dense_51_bias_v:2=
+assignvariableop_41_nadam_dense_52_kernel_v:2d7
)assignvariableop_42_nadam_dense_52_bias_v:d>
+assignvariableop_43_nadam_dense_53_kernel_v:	d?8
)assignvariableop_44_nadam_dense_53_bias_v:	??
+assignvariableop_45_nadam_dense_54_kernel_v:
??8
)assignvariableop_46_nadam_dense_54_bias_v:	?
identity_48??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	2
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
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_49_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_49_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_50_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_50_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_51_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_51_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_52_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_52_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_53_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_53_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_54_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_54_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_nadam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_nadam_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_nadam_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_nadam_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_nadam_momentum_cacheIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_nadam_gamma_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_nadam_dense_49_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_nadam_dense_49_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_50_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_50_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_nadam_dense_51_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_nadam_dense_51_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_dense_52_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_dense_52_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_53_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_53_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_54_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_54_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp!assignvariableop_34_nadam_gamma_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_nadam_dense_49_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_nadam_dense_49_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_nadam_dense_50_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_nadam_dense_50_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_nadam_dense_51_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_nadam_dense_51_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_nadam_dense_52_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_nadam_dense_52_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_nadam_dense_53_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_nadam_dense_53_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_nadam_dense_54_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_nadam_dense_54_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47?
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
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
C__inference_dense_53_layer_call_and_return_conditional_losses_41974

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
C__inference_dense_50_layer_call_and_return_conditional_losses_41115

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
:??????????2
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
:??????????
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
:??????????
2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_53_layer_call_and_return_conditional_losses_41174

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
C__inference_dense_51_layer_call_and_return_conditional_losses_41140

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
C__inference_dense_49_layer_call_and_return_conditional_losses_41078

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_8_layer_call_and_return_conditional_losses_41127

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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
(__inference_dense_52_layer_call_fn_41963

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
C__inference_dense_52_layer_call_and_return_conditional_losses_411572
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
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_41814

inputs7
%einsum_einsum_readvariableop_resource:
identity??einsum/Einsum/ReadVariableOp?
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype02
einsum/Einsum/ReadVariableOp?
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????*
equationijk,kl->ijl2
einsum/Einsum?
IdentityIdentityeinsum/Einsum:output:0^einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?`
?
__inference__traced_save_42167
file_prefix$
 savev2_gamma_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop,
(savev2_nadam_gamma_m_read_readvariableop6
2savev2_nadam_dense_49_kernel_m_read_readvariableop4
0savev2_nadam_dense_49_bias_m_read_readvariableop6
2savev2_nadam_dense_50_kernel_m_read_readvariableop4
0savev2_nadam_dense_50_bias_m_read_readvariableop6
2savev2_nadam_dense_51_kernel_m_read_readvariableop4
0savev2_nadam_dense_51_bias_m_read_readvariableop6
2savev2_nadam_dense_52_kernel_m_read_readvariableop4
0savev2_nadam_dense_52_bias_m_read_readvariableop6
2savev2_nadam_dense_53_kernel_m_read_readvariableop4
0savev2_nadam_dense_53_bias_m_read_readvariableop6
2savev2_nadam_dense_54_kernel_m_read_readvariableop4
0savev2_nadam_dense_54_bias_m_read_readvariableop,
(savev2_nadam_gamma_v_read_readvariableop6
2savev2_nadam_dense_49_kernel_v_read_readvariableop4
0savev2_nadam_dense_49_bias_v_read_readvariableop6
2savev2_nadam_dense_50_kernel_v_read_readvariableop4
0savev2_nadam_dense_50_bias_v_read_readvariableop6
2savev2_nadam_dense_51_kernel_v_read_readvariableop4
0savev2_nadam_dense_51_bias_v_read_readvariableop6
2savev2_nadam_dense_52_kernel_v_read_readvariableop4
0savev2_nadam_dense_52_bias_v_read_readvariableop6
2savev2_nadam_dense_53_kernel_v_read_readvariableop4
0savev2_nadam_dense_53_bias_v_read_readvariableop6
2savev2_nadam_dense_54_kernel_v_read_readvariableop4
0savev2_nadam_dense_54_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_gamma_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop(savev2_nadam_gamma_m_read_readvariableop2savev2_nadam_dense_49_kernel_m_read_readvariableop0savev2_nadam_dense_49_bias_m_read_readvariableop2savev2_nadam_dense_50_kernel_m_read_readvariableop0savev2_nadam_dense_50_bias_m_read_readvariableop2savev2_nadam_dense_51_kernel_m_read_readvariableop0savev2_nadam_dense_51_bias_m_read_readvariableop2savev2_nadam_dense_52_kernel_m_read_readvariableop0savev2_nadam_dense_52_bias_m_read_readvariableop2savev2_nadam_dense_53_kernel_m_read_readvariableop0savev2_nadam_dense_53_bias_m_read_readvariableop2savev2_nadam_dense_54_kernel_m_read_readvariableop0savev2_nadam_dense_54_bias_m_read_readvariableop(savev2_nadam_gamma_v_read_readvariableop2savev2_nadam_dense_49_kernel_v_read_readvariableop0savev2_nadam_dense_49_bias_v_read_readvariableop2savev2_nadam_dense_50_kernel_v_read_readvariableop0savev2_nadam_dense_50_bias_v_read_readvariableop2savev2_nadam_dense_51_kernel_v_read_readvariableop0savev2_nadam_dense_51_bias_v_read_readvariableop2savev2_nadam_dense_52_kernel_v_read_readvariableop0savev2_nadam_dense_52_bias_v_read_readvariableop2savev2_nadam_dense_53_kernel_v_read_readvariableop0savev2_nadam_dense_53_bias_v_read_readvariableop2savev2_nadam_dense_54_kernel_v_read_readvariableop0savev2_nadam_dense_54_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::
:
:
2:2:2d:d:	d?:?:
??:?: : : : : : : : ::::
:
:
2:2:2d:d:	d?:?:
??:?::::
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

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
2: 

_output_shapes
:2:$ 

_output_shapes

:2d: 	

_output_shapes
:d:%
!

_output_shapes
:	d?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
2: 

_output_shapes
:2:$ 

_output_shapes

:2d: 

_output_shapes
:d:%!

_output_shapes
:	d?:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:$# 

_output_shapes

::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:
: '

_output_shapes
:
:$( 

_output_shapes

:
2: )

_output_shapes
:2:$* 

_output_shapes

:2d: +

_output_shapes
:d:%,!

_output_shapes
:	d?:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:!/

_output_shapes	
:?:0

_output_shapes
: 
?
?
(__inference_dense_54_layer_call_fn_42003

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
C__inference_dense_54_layer_call_and_return_conditional_losses_411912
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
?

?
C__inference_dense_52_layer_call_and_return_conditional_losses_41954

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

?
#__inference_signature_wrapper_41561
input_9
unknown:
	unknown_0:
	unknown_1:
	unknown_2:

	unknown_3:

	unknown_4:
2
	unknown_5:2
	unknown_6:2d
	unknown_7:d
	unknown_8:	d?
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_410282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
? 
?
C__inference_dense_50_layer_call_and_return_conditional_losses_41892

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
:??????????2
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
:??????????
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
:??????????
2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
э
?
 __inference__wrapped_model_41028
input_9R
@model_8_fully_connected2_8_einsum_einsum_readvariableop_resource:D
2model_8_dense_49_tensordot_readvariableop_resource:>
0model_8_dense_49_biasadd_readvariableop_resource:D
2model_8_dense_50_tensordot_readvariableop_resource:
>
0model_8_dense_50_biasadd_readvariableop_resource:
A
/model_8_dense_51_matmul_readvariableop_resource:
2>
0model_8_dense_51_biasadd_readvariableop_resource:2A
/model_8_dense_52_matmul_readvariableop_resource:2d>
0model_8_dense_52_biasadd_readvariableop_resource:dB
/model_8_dense_53_matmul_readvariableop_resource:	d??
0model_8_dense_53_biasadd_readvariableop_resource:	?C
/model_8_dense_54_matmul_readvariableop_resource:
???
0model_8_dense_54_biasadd_readvariableop_resource:	?
identity??'model_8/dense_49/BiasAdd/ReadVariableOp?)model_8/dense_49/Tensordot/ReadVariableOp?'model_8/dense_50/BiasAdd/ReadVariableOp?)model_8/dense_50/Tensordot/ReadVariableOp?'model_8/dense_51/BiasAdd/ReadVariableOp?&model_8/dense_51/MatMul/ReadVariableOp?'model_8/dense_52/BiasAdd/ReadVariableOp?&model_8/dense_52/MatMul/ReadVariableOp?'model_8/dense_53/BiasAdd/ReadVariableOp?&model_8/dense_53/MatMul/ReadVariableOp?'model_8/dense_54/BiasAdd/ReadVariableOp?&model_8/dense_54/MatMul/ReadVariableOp?7model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp?
7model_8/fully_connected2_8/einsum/Einsum/ReadVariableOpReadVariableOp@model_8_fully_connected2_8_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype029
7model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp?
(model_8/fully_connected2_8/einsum/EinsumEinsuminput_9?model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:??????????*
equationijk,kl->ijl2*
(model_8/fully_connected2_8/einsum/Einsum?
model_8/tf.nn.relu_8/ReluRelu1model_8/fully_connected2_8/einsum/Einsum:output:0*
T0*,
_output_shapes
:??????????2
model_8/tf.nn.relu_8/Relu?
)model_8/dense_49/Tensordot/ReadVariableOpReadVariableOp2model_8_dense_49_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_8/dense_49/Tensordot/ReadVariableOp?
model_8/dense_49/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
model_8/dense_49/Tensordot/axes?
model_8/dense_49/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
model_8/dense_49/Tensordot/free?
 model_8/dense_49/Tensordot/ShapeShape'model_8/tf.nn.relu_8/Relu:activations:0*
T0*
_output_shapes
:2"
 model_8/dense_49/Tensordot/Shape?
(model_8/dense_49/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_8/dense_49/Tensordot/GatherV2/axis?
#model_8/dense_49/Tensordot/GatherV2GatherV2)model_8/dense_49/Tensordot/Shape:output:0(model_8/dense_49/Tensordot/free:output:01model_8/dense_49/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model_8/dense_49/Tensordot/GatherV2?
*model_8/dense_49/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_8/dense_49/Tensordot/GatherV2_1/axis?
%model_8/dense_49/Tensordot/GatherV2_1GatherV2)model_8/dense_49/Tensordot/Shape:output:0(model_8/dense_49/Tensordot/axes:output:03model_8/dense_49/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%model_8/dense_49/Tensordot/GatherV2_1?
 model_8/dense_49/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model_8/dense_49/Tensordot/Const?
model_8/dense_49/Tensordot/ProdProd,model_8/dense_49/Tensordot/GatherV2:output:0)model_8/dense_49/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
model_8/dense_49/Tensordot/Prod?
"model_8/dense_49/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model_8/dense_49/Tensordot/Const_1?
!model_8/dense_49/Tensordot/Prod_1Prod.model_8/dense_49/Tensordot/GatherV2_1:output:0+model_8/dense_49/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!model_8/dense_49/Tensordot/Prod_1?
&model_8/dense_49/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_8/dense_49/Tensordot/concat/axis?
!model_8/dense_49/Tensordot/concatConcatV2(model_8/dense_49/Tensordot/free:output:0(model_8/dense_49/Tensordot/axes:output:0/model_8/dense_49/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!model_8/dense_49/Tensordot/concat?
 model_8/dense_49/Tensordot/stackPack(model_8/dense_49/Tensordot/Prod:output:0*model_8/dense_49/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 model_8/dense_49/Tensordot/stack?
$model_8/dense_49/Tensordot/transpose	Transpose'model_8/tf.nn.relu_8/Relu:activations:0*model_8/dense_49/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2&
$model_8/dense_49/Tensordot/transpose?
"model_8/dense_49/Tensordot/ReshapeReshape(model_8/dense_49/Tensordot/transpose:y:0)model_8/dense_49/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"model_8/dense_49/Tensordot/Reshape?
!model_8/dense_49/Tensordot/MatMulMatMul+model_8/dense_49/Tensordot/Reshape:output:01model_8/dense_49/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_8/dense_49/Tensordot/MatMul?
"model_8/dense_49/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model_8/dense_49/Tensordot/Const_2?
(model_8/dense_49/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_8/dense_49/Tensordot/concat_1/axis?
#model_8/dense_49/Tensordot/concat_1ConcatV2,model_8/dense_49/Tensordot/GatherV2:output:0+model_8/dense_49/Tensordot/Const_2:output:01model_8/dense_49/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_8/dense_49/Tensordot/concat_1?
model_8/dense_49/TensordotReshape+model_8/dense_49/Tensordot/MatMul:product:0,model_8/dense_49/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
model_8/dense_49/Tensordot?
'model_8/dense_49/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_8/dense_49/BiasAdd/ReadVariableOp?
model_8/dense_49/BiasAddBiasAdd#model_8/dense_49/Tensordot:output:0/model_8/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_8/dense_49/BiasAdd?
model_8/dense_49/ReluRelu!model_8/dense_49/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_8/dense_49/Relu?
)model_8/dense_50/Tensordot/ReadVariableOpReadVariableOp2model_8_dense_50_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype02+
)model_8/dense_50/Tensordot/ReadVariableOp?
model_8/dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
model_8/dense_50/Tensordot/axes?
model_8/dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
model_8/dense_50/Tensordot/free?
 model_8/dense_50/Tensordot/ShapeShape#model_8/dense_49/Relu:activations:0*
T0*
_output_shapes
:2"
 model_8/dense_50/Tensordot/Shape?
(model_8/dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_8/dense_50/Tensordot/GatherV2/axis?
#model_8/dense_50/Tensordot/GatherV2GatherV2)model_8/dense_50/Tensordot/Shape:output:0(model_8/dense_50/Tensordot/free:output:01model_8/dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model_8/dense_50/Tensordot/GatherV2?
*model_8/dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_8/dense_50/Tensordot/GatherV2_1/axis?
%model_8/dense_50/Tensordot/GatherV2_1GatherV2)model_8/dense_50/Tensordot/Shape:output:0(model_8/dense_50/Tensordot/axes:output:03model_8/dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%model_8/dense_50/Tensordot/GatherV2_1?
 model_8/dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model_8/dense_50/Tensordot/Const?
model_8/dense_50/Tensordot/ProdProd,model_8/dense_50/Tensordot/GatherV2:output:0)model_8/dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
model_8/dense_50/Tensordot/Prod?
"model_8/dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model_8/dense_50/Tensordot/Const_1?
!model_8/dense_50/Tensordot/Prod_1Prod.model_8/dense_50/Tensordot/GatherV2_1:output:0+model_8/dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!model_8/dense_50/Tensordot/Prod_1?
&model_8/dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_8/dense_50/Tensordot/concat/axis?
!model_8/dense_50/Tensordot/concatConcatV2(model_8/dense_50/Tensordot/free:output:0(model_8/dense_50/Tensordot/axes:output:0/model_8/dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!model_8/dense_50/Tensordot/concat?
 model_8/dense_50/Tensordot/stackPack(model_8/dense_50/Tensordot/Prod:output:0*model_8/dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 model_8/dense_50/Tensordot/stack?
$model_8/dense_50/Tensordot/transpose	Transpose#model_8/dense_49/Relu:activations:0*model_8/dense_50/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2&
$model_8/dense_50/Tensordot/transpose?
"model_8/dense_50/Tensordot/ReshapeReshape(model_8/dense_50/Tensordot/transpose:y:0)model_8/dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"model_8/dense_50/Tensordot/Reshape?
!model_8/dense_50/Tensordot/MatMulMatMul+model_8/dense_50/Tensordot/Reshape:output:01model_8/dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!model_8/dense_50/Tensordot/MatMul?
"model_8/dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
2$
"model_8/dense_50/Tensordot/Const_2?
(model_8/dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_8/dense_50/Tensordot/concat_1/axis?
#model_8/dense_50/Tensordot/concat_1ConcatV2,model_8/dense_50/Tensordot/GatherV2:output:0+model_8/dense_50/Tensordot/Const_2:output:01model_8/dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#model_8/dense_50/Tensordot/concat_1?
model_8/dense_50/TensordotReshape+model_8/dense_50/Tensordot/MatMul:product:0,model_8/dense_50/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????
2
model_8/dense_50/Tensordot?
'model_8/dense_50/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_50_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_8/dense_50/BiasAdd/ReadVariableOp?
model_8/dense_50/BiasAddBiasAdd#model_8/dense_50/Tensordot:output:0/model_8/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
model_8/dense_50/BiasAdd?
model_8/dense_50/ReluRelu!model_8/dense_50/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
2
model_8/dense_50/Relu?
&model_8/lambda_8/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_8/lambda_8/Min/reduction_indices?
model_8/lambda_8/MinMin#model_8/dense_50/Relu:activations:0/model_8/lambda_8/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????
2
model_8/lambda_8/Min?
&model_8/dense_51/MatMul/ReadVariableOpReadVariableOp/model_8_dense_51_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype02(
&model_8/dense_51/MatMul/ReadVariableOp?
model_8/dense_51/MatMulMatMulmodel_8/lambda_8/Min:output:0.model_8/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_8/dense_51/MatMul?
'model_8/dense_51/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_51_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'model_8/dense_51/BiasAdd/ReadVariableOp?
model_8/dense_51/BiasAddBiasAdd!model_8/dense_51/MatMul:product:0/model_8/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_8/dense_51/BiasAdd?
model_8/dense_51/ReluRelu!model_8/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_8/dense_51/Relu?
&model_8/dense_52/MatMul/ReadVariableOpReadVariableOp/model_8_dense_52_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02(
&model_8/dense_52/MatMul/ReadVariableOp?
model_8/dense_52/MatMulMatMul#model_8/dense_51/Relu:activations:0.model_8/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_8/dense_52/MatMul?
'model_8/dense_52/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_52_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'model_8/dense_52/BiasAdd/ReadVariableOp?
model_8/dense_52/BiasAddBiasAdd!model_8/dense_52/MatMul:product:0/model_8/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_8/dense_52/BiasAdd?
model_8/dense_52/ReluRelu!model_8/dense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
model_8/dense_52/Relu?
&model_8/dense_53/MatMul/ReadVariableOpReadVariableOp/model_8_dense_53_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype02(
&model_8/dense_53/MatMul/ReadVariableOp?
model_8/dense_53/MatMulMatMul#model_8/dense_52/Relu:activations:0.model_8/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_53/MatMul?
'model_8/dense_53/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_8/dense_53/BiasAdd/ReadVariableOp?
model_8/dense_53/BiasAddBiasAdd!model_8/dense_53/MatMul:product:0/model_8/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_53/BiasAdd?
model_8/dense_53/ReluRelu!model_8/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_8/dense_53/Relu?
&model_8/dense_54/MatMul/ReadVariableOpReadVariableOp/model_8_dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_8/dense_54/MatMul/ReadVariableOp?
model_8/dense_54/MatMulMatMul#model_8/dense_53/Relu:activations:0.model_8/dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_54/MatMul?
'model_8/dense_54/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_8/dense_54/BiasAdd/ReadVariableOp?
model_8/dense_54/BiasAddBiasAdd!model_8/dense_54/MatMul:product:0/model_8/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_54/BiasAdd?
model_8/dense_54/SigmoidSigmoid!model_8/dense_54/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_8/dense_54/Sigmoid?
IdentityIdentitymodel_8/dense_54/Sigmoid:y:0(^model_8/dense_49/BiasAdd/ReadVariableOp*^model_8/dense_49/Tensordot/ReadVariableOp(^model_8/dense_50/BiasAdd/ReadVariableOp*^model_8/dense_50/Tensordot/ReadVariableOp(^model_8/dense_51/BiasAdd/ReadVariableOp'^model_8/dense_51/MatMul/ReadVariableOp(^model_8/dense_52/BiasAdd/ReadVariableOp'^model_8/dense_52/MatMul/ReadVariableOp(^model_8/dense_53/BiasAdd/ReadVariableOp'^model_8/dense_53/MatMul/ReadVariableOp(^model_8/dense_54/BiasAdd/ReadVariableOp'^model_8/dense_54/MatMul/ReadVariableOp8^model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 2R
'model_8/dense_49/BiasAdd/ReadVariableOp'model_8/dense_49/BiasAdd/ReadVariableOp2V
)model_8/dense_49/Tensordot/ReadVariableOp)model_8/dense_49/Tensordot/ReadVariableOp2R
'model_8/dense_50/BiasAdd/ReadVariableOp'model_8/dense_50/BiasAdd/ReadVariableOp2V
)model_8/dense_50/Tensordot/ReadVariableOp)model_8/dense_50/Tensordot/ReadVariableOp2R
'model_8/dense_51/BiasAdd/ReadVariableOp'model_8/dense_51/BiasAdd/ReadVariableOp2P
&model_8/dense_51/MatMul/ReadVariableOp&model_8/dense_51/MatMul/ReadVariableOp2R
'model_8/dense_52/BiasAdd/ReadVariableOp'model_8/dense_52/BiasAdd/ReadVariableOp2P
&model_8/dense_52/MatMul/ReadVariableOp&model_8/dense_52/MatMul/ReadVariableOp2R
'model_8/dense_53/BiasAdd/ReadVariableOp'model_8/dense_53/BiasAdd/ReadVariableOp2P
&model_8/dense_53/MatMul/ReadVariableOp&model_8/dense_53/MatMul/ReadVariableOp2R
'model_8/dense_54/BiasAdd/ReadVariableOp'model_8/dense_54/BiasAdd/ReadVariableOp2P
&model_8/dense_54/MatMul/ReadVariableOp&model_8/dense_54/MatMul/ReadVariableOp2r
7model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp7model_8/fully_connected2_8/einsum/Einsum/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_9
?
?
(__inference_dense_53_layer_call_fn_41983

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
C__inference_dense_53_layer_call_and_return_conditional_losses_411742
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
?
_
C__inference_lambda_8_layer_call_and_return_conditional_losses_41281

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
:??????????
:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_41191

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
?
?
(__inference_dense_50_layer_call_fn_41901

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
:??????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_411152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_model_8_layer_call_fn_41807

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:

	unknown_3:

	unknown_4:
2
	unknown_5:2
	unknown_6:2d
	unknown_7:d
	unknown_8:	d?
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_413842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
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
input_95
serving_default_input_9:0??????????=
dense_541
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?/
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
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?+
_tf_keras_network?+{"name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "FullyConnected2", "config": {"layer was saved without config": true}, "name": "fully_connected2_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_8", "inbound_nodes": [["fully_connected2_8", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_49", "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_50", "inbound_nodes": [[["dense_49", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_8", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMQAAAAdABqAWoCfABkAWQCjQJTACkDTukBAAAAKQHa\nBGF4aXMpA9oCdGbaBG1hdGjaCnJlZHVjZV9taW4pAdoBeKkAcgcAAAD6HzxpcHl0aG9uLWlucHV0\nLTI2LThjZGM3Y2JlYWUxOD7aCDxsYW1iZGE+FwAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAADAAAAQwAAAHMQAAAAfABkARkAfABkAhkAZgJTACkDTukAAAAA6QIA\nAACpACkB2gVzaGFwZXIDAAAAcgMAAAD6HzxpcHl0aG9uLWlucHV0LTI2LThjZGM3Y2JlYWUxOD7a\nCDxsYW1iZGE+GAAAAHMCAAAAAAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda_8", "inbound_nodes": [[["dense_50", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_51", "inbound_nodes": [[["lambda_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_52", "inbound_nodes": [[["dense_51", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_53", "inbound_nodes": [[["dense_52", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 2500, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_54", "inbound_nodes": [[["dense_53", 0, 0, {}]]]}], "input_layers": [["input_9", 0, 0]], "output_layers": [["dense_54", 0, 0]]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 600, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 600, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 600, 2]}, "float32", "input_9"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.004999999888241291, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}
?
	gamma
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "fully_connected2_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "FullyConnected2", "config": {"layer was saved without config": true}}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.nn.relu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "inbound_nodes": [["fully_connected2_8", 0, 0, {"name": null}]], "shared_object_id": 1}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600, 30]}}
?	

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_49", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600, 20]}}
?	
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "lambda_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_8", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMQAAAAdABqAWoCfABkAWQCjQJTACkDTukBAAAAKQHa\nBGF4aXMpA9oCdGbaBG1hdGjaCnJlZHVjZV9taW4pAdoBeKkAcgcAAAD6HzxpcHl0aG9uLWlucHV0\nLTI2LThjZGM3Y2JlYWUxOD7aCDxsYW1iZGE+FwAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAADAAAAQwAAAHMQAAAAfABkARkAfABkAhkAZgJTACkDTukAAAAA6QIA\nAACpACkB2gVzaGFwZXIDAAAAcgMAAAD6HzxpcHl0aG9uLWlucHV0LTI2LThjZGM3Y2JlYWUxOD7a\nCDxsYW1iZGE+GAAAAHMCAAAAAAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "inbound_nodes": [[["dense_50", 0, 0, {}]]], "shared_object_id": 8}
?	

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lambda_8", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?	

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_51", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?	

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_52", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?	

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 2500, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_53", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate
Dmomentum_cachemwmxmymzm{'m|(m}-m~.m3m?4m?9m?:m?v?v?v?v?v?'v?(v?-v?.v?3v?4v?9v?:v?"
	optimizer
~
0
1
2
3
4
'5
(6
-7
.8
39
410
911
:12"
trackable_list_wrapper
~
0
1
2
3
4
'5
(6
-7
.8
39
410
911
:12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Elayer_metrics
Fmetrics

Glayers
Hnon_trainable_variables
Ilayer_regularization_losses
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:2gamma
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Jlayer_metrics
Kmetrics

Llayers
Mnon_trainable_variables
Nlayer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
!:2dense_49/kernel
:2dense_49/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Olayer_metrics
Pmetrics

Qlayers
Rnon_trainable_variables
Slayer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_50/kernel
:
2dense_50/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
 	variables
Tlayer_metrics
Umetrics

Vlayers
Wnon_trainable_variables
Xlayer_regularization_losses
!regularization_losses
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
#trainable_variables
$	variables
Ylayer_metrics
Zmetrics

[layers
\non_trainable_variables
]layer_regularization_losses
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
22dense_51/kernel
:22dense_51/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables
*	variables
^layer_metrics
_metrics

`layers
anon_trainable_variables
blayer_regularization_losses
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2d2dense_52/kernel
:d2dense_52/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/trainable_variables
0	variables
clayer_metrics
dmetrics

elayers
fnon_trainable_variables
glayer_regularization_losses
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	d?2dense_53/kernel
:?2dense_53/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5trainable_variables
6	variables
hlayer_metrics
imetrics

jlayers
knon_trainable_variables
llayer_regularization_losses
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_54/kernel
:?2dense_54/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;trainable_variables
<	variables
mlayer_metrics
nmetrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
=regularization_losses
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
trackable_dict_wrapper
'
r0"
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
	stotal
	tcount
u	variables
v	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 29}
:  (2total
:  (2count
.
s0
t1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:2Nadam/gamma/m
':%2Nadam/dense_49/kernel/m
!:2Nadam/dense_49/bias/m
':%
2Nadam/dense_50/kernel/m
!:
2Nadam/dense_50/bias/m
':%
22Nadam/dense_51/kernel/m
!:22Nadam/dense_51/bias/m
':%2d2Nadam/dense_52/kernel/m
!:d2Nadam/dense_52/bias/m
(:&	d?2Nadam/dense_53/kernel/m
": ?2Nadam/dense_53/bias/m
):'
??2Nadam/dense_54/kernel/m
": ?2Nadam/dense_54/bias/m
:2Nadam/gamma/v
':%2Nadam/dense_49/kernel/v
!:2Nadam/dense_49/bias/v
':%
2Nadam/dense_50/kernel/v
!:
2Nadam/dense_50/bias/v
':%
22Nadam/dense_51/kernel/v
!:22Nadam/dense_51/bias/v
':%2d2Nadam/dense_52/kernel/v
!:d2Nadam/dense_52/bias/v
(:&	d?2Nadam/dense_53/kernel/v
": ?2Nadam/dense_53/bias/v
):'
??2Nadam/dense_54/kernel/v
": ?2Nadam/dense_54/bias/v
?2?
B__inference_model_8_layer_call_and_return_conditional_losses_41653
B__inference_model_8_layer_call_and_return_conditional_losses_41745
B__inference_model_8_layer_call_and_return_conditional_losses_41483
B__inference_model_8_layer_call_and_return_conditional_losses_41522?
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
 __inference__wrapped_model_41028?
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
input_9??????????
?2?
'__inference_model_8_layer_call_fn_41227
'__inference_model_8_layer_call_fn_41776
'__inference_model_8_layer_call_fn_41807
'__inference_model_8_layer_call_fn_41444?
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
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_41814?
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
2__inference_fully_connected2_8_layer_call_fn_41821?
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
C__inference_dense_49_layer_call_and_return_conditional_losses_41852?
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
(__inference_dense_49_layer_call_fn_41861?
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
C__inference_dense_50_layer_call_and_return_conditional_losses_41892?
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
(__inference_dense_50_layer_call_fn_41901?
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
C__inference_lambda_8_layer_call_and_return_conditional_losses_41907
C__inference_lambda_8_layer_call_and_return_conditional_losses_41913?
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
(__inference_lambda_8_layer_call_fn_41918
(__inference_lambda_8_layer_call_fn_41923?
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
C__inference_dense_51_layer_call_and_return_conditional_losses_41934?
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
(__inference_dense_51_layer_call_fn_41943?
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
C__inference_dense_52_layer_call_and_return_conditional_losses_41954?
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
(__inference_dense_52_layer_call_fn_41963?
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
C__inference_dense_53_layer_call_and_return_conditional_losses_41974?
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
(__inference_dense_53_layer_call_fn_41983?
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
C__inference_dense_54_layer_call_and_return_conditional_losses_41994?
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
(__inference_dense_54_layer_call_fn_42003?
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
#__inference_signature_wrapper_41561input_9"?
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
 __inference__wrapped_model_41028|'(-.349:5?2
+?(
&?#
input_9??????????
? "4?1
/
dense_54#? 
dense_54???????????
C__inference_dense_49_layer_call_and_return_conditional_losses_41852f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
(__inference_dense_49_layer_call_fn_41861Y4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_dense_50_layer_call_and_return_conditional_losses_41892f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????

? ?
(__inference_dense_50_layer_call_fn_41901Y4?1
*?'
%?"
inputs??????????
? "???????????
?
C__inference_dense_51_layer_call_and_return_conditional_losses_41934\'(/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????2
? {
(__inference_dense_51_layer_call_fn_41943O'(/?,
%?"
 ?
inputs?????????

? "??????????2?
C__inference_dense_52_layer_call_and_return_conditional_losses_41954\-./?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????d
? {
(__inference_dense_52_layer_call_fn_41963O-./?,
%?"
 ?
inputs?????????2
? "??????????d?
C__inference_dense_53_layer_call_and_return_conditional_losses_41974]34/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? |
(__inference_dense_53_layer_call_fn_41983P34/?,
%?"
 ?
inputs?????????d
? "????????????
C__inference_dense_54_layer_call_and_return_conditional_losses_41994^9:0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_54_layer_call_fn_42003Q9:0?-
&?#
!?
inputs??????????
? "????????????
M__inference_fully_connected2_8_layer_call_and_return_conditional_losses_41814e4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
2__inference_fully_connected2_8_layer_call_fn_41821X4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_lambda_8_layer_call_and_return_conditional_losses_41907e<?9
2?/
%?"
inputs??????????


 
p 
? "%?"
?
0?????????

? ?
C__inference_lambda_8_layer_call_and_return_conditional_losses_41913e<?9
2?/
%?"
inputs??????????


 
p
? "%?"
?
0?????????

? ?
(__inference_lambda_8_layer_call_fn_41918X<?9
2?/
%?"
inputs??????????


 
p 
? "??????????
?
(__inference_lambda_8_layer_call_fn_41923X<?9
2?/
%?"
inputs??????????


 
p
? "??????????
?
B__inference_model_8_layer_call_and_return_conditional_losses_41483v'(-.349:=?:
3?0
&?#
input_9??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_41522v'(-.349:=?:
3?0
&?#
input_9??????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_41653u'(-.349:<?9
2?/
%?"
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_41745u'(-.349:<?9
2?/
%?"
inputs??????????
p

 
? "&?#
?
0??????????
? ?
'__inference_model_8_layer_call_fn_41227i'(-.349:=?:
3?0
&?#
input_9??????????
p 

 
? "????????????
'__inference_model_8_layer_call_fn_41444i'(-.349:=?:
3?0
&?#
input_9??????????
p

 
? "????????????
'__inference_model_8_layer_call_fn_41776h'(-.349:<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
'__inference_model_8_layer_call_fn_41807h'(-.349:<?9
2?/
%?"
inputs??????????
p

 
? "????????????
#__inference_signature_wrapper_41561?'(-.349:@?=
? 
6?3
1
input_9&?#
input_9??????????"4?1
/
dense_54#? 
dense_54??????????