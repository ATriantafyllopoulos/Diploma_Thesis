typedef struct primitive
{
	primitive *left;
	primitive *right;
	//primitive *leftmost;
	//primitive *rightmost;
	primitive *parent;

	//change range of leaves to indices
	//no need to fetch pointer form memory
	int leftmost;
	int rightmost;
	//trying to save collision using indices
	primitive *collisions[8];

	unsigned int collisionCounter; //this must not exceed 7

	float radius;

	int id;
	bool isLeaf;
	bool keepMoving;
	float mass;
	//state vector
	float3 centroid;
	float3 linearMomentum;
	//no need for rotation yet
	//we use primitive spheres
	//float3 angularMomentum;
	//float4 quaternion;
	//added inequality operator overload
	//might be useful when incorporating radix sort
	//remove otherwise
	//bool operator != (primitive &rhs) { return &*this == &rhs; }
} Primitive;

//function overload also used during radix sort
//if not needed then remove
//int CoutCast(primitive val) { return 0; }

