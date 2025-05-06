def trilevel_z1(x):
	return (x[0] + x[1] + 2*x[2] + 4) * (-x[0] - x[1] + x[2] + 2*x[3] + 1)
def constraint_violations(x):
	violations = 0

	v1 = -3 * x[0] + 7*x[1] + x[2] + x[4]
	if v1 != 10:
		violations += 1
	
	v2  = 14*x[0] + 4*x[1] + x[5]
	if v2 != 6:
		violations += 1
	
	v3 = x[0] + x[1] + x[2] - x[3] + x[6]
	if v3 != 5:
		violations += 1
	
	v4 = 2*x[0] + x[1] + 2*x[3] + x[7]
	if v4 != 8:
		violations += 1
	
	return violations
def trilevel_z2(x):
	return 2*x[1] + x[2] + 3*x[3]

def trilevel_z3(x):
	a = 2 * x[0] + 3 * x[1] + 2 * x[2] - 3 * x[3]
	b = 5 * x[0] + 11 * x[1] + x[4] + 29
	return a/b
