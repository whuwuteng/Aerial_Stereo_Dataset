/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef COST_AGGREGATION_H_
#define COST_AGGREGATION_H_

#define ITER_COPY			0
#define ITER_NORMAL			1

#define MIN_COMPUTE			0
#define MIN_NOCOMPUTE		1

#define DIR_UPDOWN			0
#define DIR_DOWNUP			1
#define DIR_LEFTRIGHT		2
#define DIR_RIGHTLEFT		3

template<int add_col, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGenericIndexesIncrement(int *index, int *index_im, int *col, const int add_index, const int add_imindex) {
	*index += add_index;
	if(recompute || join_dispcomputation) {
		*index_im += add_imindex;
		if(recompute) {
			*col += add_col;
		}
	}
}

template<int add_index, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationDiagonalGenericIndexesIncrement(int *index, int *index_im, int *col, const int cols, const int initial_row, const int i, const int dis) {
	*col += add_index;
	if(add_index > 0 && *col >= cols) {
		*col = 0;
	} else if(*col < 0) {
		*col = cols-1;
	}
	*index = abs(initial_row-i)*cols*MAX_DISPARITY+*col*MAX_DISPARITY+dis;
	if(recompute || join_dispcomputation) {
		*index_im = abs(initial_row-i)*cols+*col;
	}
}

// iter_type ITER_NORMAL or ITER_COPY
// this is the kernl
template<class T, int iter_type, int min_type, int dir_type, bool first_iteration, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGenericIteration(int index, int index_im, int col, uint32_t *old_values, uint32_t *old_values2, int *old_value1, int *old_value2, int *old_value3, int *old_value4, int *old_value5, int *old_value6, int *old_value7, int *old_value8, uint32_t *min_cost, uint32_t *min_cost_p2, uint8_t* d_cost, uint8_t *d_L, const int p1_vector, const int p2_vector, const T *_d_transform0, const T *_d_transform1, const int lane, const int MAX_PAD, const int dis, T *rp0, T *rp1, T *rp2, T *rp3, T *rp4, T *rp5, T *rp6, T *rp7, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const T __restrict__ *d_transform0 = _d_transform0;
	const T __restrict__ *d_transform1 = _d_transform1;
	uint32_t costs, next_dis, prev_dis;
	uint32_t costs2, next_dis2, prev_dis2;

	if(iter_type == ITER_NORMAL) {
		// First shuffle
		// read thread+1 *old_value4
		int prev_dis1 = shfl_up_32(*old_value8, 1);
		if(lane == 0) {
			prev_dis1 = MAX_PAD;
		}

		// Second shuffle
		// read thread-1 *old_value1
		int next_dis8 = shfl_down_32(*old_value1, 1);
		if(lane == 31) {
			next_dis8 = MAX_PAD;
		}

		// Shift + rotate
		//next_dis = __funnelshift_r(next_dis4, *old_values, 8);
		next_dis = __byte_perm(*old_values2, next_dis8, 0x4321);
		prev_dis = __byte_perm(*old_values, prev_dis1, 0x2104);

		next_dis = next_dis + p1_vector;
		prev_dis = prev_dis + p1_vector;
	}

	// compute cost again, strange?
	// recomputer cost is not correct for disparity
	if(recompute) {
		const int dif = col - dis;
		if(dir_type == DIR_LEFTRIGHT) {
			//if(lane == 0) {
			// initial
			if(first_iteration){
				// lane = 0 is dis = 0, no need to subtract dis
				*rp3 = d_transform1[index_im];
				*rp2 = d_transform1[index_im];
				*rp1 = d_transform1[index_im];
				*rp0 = d_transform1[index_im];

				*rp7 = d_transform1[index_im];
				*rp6 = d_transform1[index_im];
				*rp5 = d_transform1[index_im];
				*rp4 = d_transform1[index_im];		
			}
			else{
				if (dif >= 0){
					*rp0 = d_transform1[index_im-dis];
				}
				else{
					*rp0 = d_transform1[index_im];
				}
			}
		} else if(dir_type == DIR_RIGHTLEFT) {
			// First iteration, load D pixels
			// initial
			if(first_iteration) {
				// combine with 4 int
				// not use the thread id
				// batch 4
				// -3~0
				if (dif >= 3){
					const uint4 right = reinterpret_cast<const uint4*>(&d_transform1[index_im-dis-3])[0];
					*rp3 = right.x;
					//printf("uint4: %d  origin: %d\n", *rp3, d_transform1[index_im-dis-3]);
					*rp2 = right.y;
					*rp1 = right.z;
					*rp0 = right.w;
				}
				else{
					*rp3 = d_transform1[index_im];
					*rp2 = d_transform1[index_im];
					*rp1 = d_transform1[index_im];
					*rp0 = d_transform1[index_im];
				}
				
				if (dif >= 7){
					const uint4 right2 = reinterpret_cast<const uint4*>(&d_transform1[index_im-dis-7])[0];
					*rp7 = right2.x;
					*rp6 = right2.y;
					*rp5 = right2.z;
					*rp4 = right2.w;
				}
				else{
					*rp7 = d_transform1[index_im];
					*rp6 = d_transform1[index_im];
					*rp5 = d_transform1[index_im];
					*rp4 = d_transform1[index_im];
				}
			}
			else{
				if (dif >= 7){
					*rp7 = d_transform1[index_im-dis-7];
				}
				else if (dif >= 0){
					*rp7 = d_transform1[index_im-dis];
				}
				else{
					*rp7 = d_transform1[index_im];
				}
			} 
			/*else if(lane == 31 && dif >= 7) {
				//?
				//*rp3 = d_transform1[index_im-dis-3];
				*rp7 = d_transform1[index_im-dis-7];
			}*/
		} else {
	/*
			__shared__ T right_p[MAX_DISPARITY+32];
			const int warp_id = threadIdx.x / WARP_SIZE;
			if(warp_id < 5) {
				const int block_imindex = index_im - warp_id + 32;
				const int rp_index = warp_id*WARP_SIZE+lane;
				const int col_cpy = col-warp_id+32;
				right_p[rp_index] = ((col_cpy-(159-rp_index)) >= 0) ? ld_gbl_cs(&d_transform1[block_imindex-(159-rp_index)]) : 0;
			}*/

			// not used
			printf("running...\n");
			// threadIdx.x [0~64)
			__shared__ T right_p[MAX_DISPARITY+32];
			const int warp_id = threadIdx.x / WARP_SIZE;
			// 2?
			const int block_imindex = index_im - warp_id + 2;
			// rp_index =  threadIdx.x
			const int rp_index = warp_id*WARP_SIZE+lane;
			const int col_cpy = col-warp_id+2;
			right_p[rp_index] = ((col_cpy-(MAX_DISPARITY_1-rp_index)) >= 0) ? d_transform1[block_imindex-(MAX_DISPARITY_1-rp_index)] : 0;
			// 64 is the threadix max
			right_p[rp_index+64] = ((col_cpy-(MAX_DISPARITY_1-rp_index-64)) >= 0) ? d_transform1[block_imindex-(MAX_DISPARITY_1-rp_index-64)] : 0;
			//right_p[rp_index+128] = ld_gbl_cs(&d_transform1[block_imindex-(129-rp_index-128)]);
			if(warp_id == 0) {
				right_p[MAX_DISPARITY+lane] = ld_gbl_cs(&d_transform1[block_imindex-(MAX_DISPARITY_1-lane)]);
			}
			__syncthreads();

			// not used
			const int px = MAX_DISPARITY+warp_id-dis-1;
			*rp0 = right_p[px];
			*rp1 = right_p[px-1];
			*rp2 = right_p[px-2];
			*rp3 = right_p[px-3];
			*rp4 = right_p[px-4];
			*rp5 = right_p[px-5];
			*rp6 = right_p[px-6];
			*rp7 = right_p[px-7];
		}
		const T left_pixel = d_transform0[index_im];
		// ^ same is 0, different is 1
		// popcount count 1
		// hamming cost
		*old_value1 = popcount(left_pixel ^ *rp0);
		//printf("computer: %d  origin: %d\n", *old_value1, d_cost[index]);
		*old_value2 = popcount(left_pixel ^ *rp1);
		//printf("computer2: %d  origin: %d\n", *old_value2, d_cost[index + 1]);
		*old_value3 = popcount(left_pixel ^ *rp2);
		*old_value4 = popcount(left_pixel ^ *rp3);
		*old_value5 = popcount(left_pixel ^ *rp4);
		*old_value6 = popcount(left_pixel ^ *rp5);
		*old_value7 = popcount(left_pixel ^ *rp6);
		*old_value8 = popcount(left_pixel ^ *rp7);
		//printf("computer8: %d  origin: %d\n", *old_value8, d_cost[index + 7]);
		//printf("value8: %d\n", *old_value8);
		if(iter_type == ITER_COPY) {
			// refer to https://forums.developer.nvidia.com/t/missing-cuda-inline-ptx-constraint-letter-for-8-bit/40427
			// *old_value4 is the first in *old_values
			*old_values = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
			*old_values2 = uchars_to_uint32(*old_value5, *old_value6, *old_value7, *old_value8);
			/**old_values = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
			*old_values2 = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index + 4]));*/
			/*if (dir_type == DIR_RIGHTLEFT){
				uint32_t test = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
				printf("computer: %d  origin: %d\n", *old_values, test);
				uint32_t test2 = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index + 4]));
				printf("computer2: %d origin2: %d\n", *old_values2, test2);
			}*/
		} else {
			costs = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
			costs2 = uchars_to_uint32(*old_value5, *old_value6, *old_value7, *old_value8);
			/*costs = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
			costs2 = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index + 4]));*/
		}
		// Prepare for next iteration
		// no need share value in one path
		/*if(dir_type == DIR_LEFTRIGHT) {
			//*rp3 = shfl_up_32(*rp3, 1);
			*rp7 = shfl_up_32(*rp7, 1);
		} else if(dir_type == DIR_RIGHTLEFT) {
			*rp0 = shfl_down_32(*rp0, 1);
		}*/
	} else {
		//printf("running 2...\n");
		// handle 8 disparity
		if(iter_type == ITER_COPY) {
			// fix 4 disparity
			*old_values = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
			*old_values2 = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index + 4]));
		} else {
			// fix 4 disparity
			costs = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
			costs2 = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index + 4]));
		}
	}

	// handle 8 disparity
	if(iter_type == ITER_NORMAL) {
		// this is important for the batch size 8
		next_dis2 = __byte_perm(*old_values, *old_value5, 0x4321);
		prev_dis2 = __byte_perm(*old_values2, *old_value4, 0x2104);

		next_dis2 = next_dis2 + p1_vector;
		prev_dis2 = prev_dis2 + p1_vector;

		// min in 4 byte
		const uint32_t min1 = __vminu4(*old_values, prev_dis);
		const uint32_t min2 = __vminu4(next_dis2, *min_cost_p2);
		const uint32_t min_prev = __vminu4(min1, min2);
		*old_values = costs + (min_prev - *min_cost);

		const uint32_t min3 = __vminu4(*old_values2, prev_dis2);
		const uint32_t min4 = __vminu4(next_dis, *min_cost_p2);
		const uint32_t min_prev2 = __vminu4(min3, min4);
		*old_values2 = costs2 + (min_prev2 - *min_cost);
	}

	// handle 8 disparity
	if(iter_type == ITER_NORMAL || !recompute) {
		uint32_to_uchars(*old_values, old_value1, old_value2, old_value3, old_value4);
		uint32_to_uchars(*old_values2, old_value5, old_value6, old_value7, old_value8);
	}

	// handle 8 disparity
	// final result
	if(join_dispcomputation) {
		const uint32_t L0_costs = *((uint32_t*) (d_L0+index));
		const uint32_t L0_costs2 = *((uint32_t*) (d_L0+index + 4));
		const uint32_t L1_costs = *((uint32_t*) (d_L1+index));
		const uint32_t L1_costs2 = *((uint32_t*) (d_L1+index + 4));
		const uint32_t L2_costs = *((uint32_t*) (d_L2+index));
		const uint32_t L2_costs2 = *((uint32_t*) (d_L2+index + 4));
		#if PATH_AGGREGATION == 8
			const uint32_t L3_costs = *((uint32_t*) (d_L3+index));
			const uint32_t L3_costs2 = *((uint32_t*) (d_L3+index + 4));
			const uint32_t L4_costs = *((uint32_t*) (d_L4+index));
			const uint32_t L4_costs2 = *((uint32_t*) (d_L4+index + 4));
			const uint32_t L5_costs = *((uint32_t*) (d_L5+index));
			const uint32_t L5_costs2 = *((uint32_t*) (d_L5+index + 4));
			const uint32_t L6_costs = *((uint32_t*) (d_L6+index));
			const uint32_t L6_costs2 = *((uint32_t*) (d_L6+index + 4));
		#endif

		int l0_x, l0_y, l0_z, l0_w;
		int l0_x2, l0_y2, l0_z2, l0_w2;
		int l1_x, l1_y, l1_z, l1_w;
		int l1_x2, l1_y2, l1_z2, l1_w2;
		int l2_x, l2_y, l2_z, l2_w;
		int l2_x2, l2_y2, l2_z2, l2_w2;
		#if PATH_AGGREGATION == 8
			int l3_x, l3_y, l3_z, l3_w;
			int l3_x2, l3_y2, l3_z2, l3_w2;
			int l4_x, l4_y, l4_z, l4_w;
			int l4_x2, l4_y2, l4_z2, l4_w2;
			int l5_x, l5_y, l5_z, l5_w;
			int l5_x2, l5_y2, l5_z2, l5_w2;
			int l6_x, l6_y, l6_z, l6_w;
			int l6_x2, l6_y2, l6_z2, l6_w2;
		#endif

		uint32_to_uchars(L0_costs, &l0_x, &l0_y, &l0_z, &l0_w);
		uint32_to_uchars(L0_costs2, &l0_x2, &l0_y2, &l0_z2, &l0_w2);
		uint32_to_uchars(L1_costs, &l1_x, &l1_y, &l1_z, &l1_w);
		uint32_to_uchars(L1_costs2, &l1_x2, &l1_y2, &l1_z2, &l1_w2);
		uint32_to_uchars(L2_costs, &l2_x, &l2_y, &l2_z, &l2_w);
		uint32_to_uchars(L2_costs2, &l2_x2, &l2_y2, &l2_z2, &l2_w2);
		#if PATH_AGGREGATION == 8
			uint32_to_uchars(L3_costs, &l3_x, &l3_y, &l3_z, &l3_w);
			uint32_to_uchars(L3_costs2, &l3_x2, &l3_y2, &l3_z2, &l3_w2);
			uint32_to_uchars(L4_costs, &l4_x, &l4_y, &l4_z, &l4_w);
			uint32_to_uchars(L4_costs2, &l4_x2, &l4_y2, &l4_z2, &l4_w2);
			uint32_to_uchars(L5_costs, &l5_x, &l5_y, &l5_z, &l5_w);
			uint32_to_uchars(L5_costs2, &l5_x2, &l5_y2, &l5_z2, &l5_w2);
			uint32_to_uchars(L6_costs, &l6_x, &l6_y, &l6_z, &l6_w);
			uint32_to_uchars(L6_costs2, &l6_x2, &l6_y2, &l6_z2, &l6_w2);
		#endif

		#if PATH_AGGREGATION == 8
			const uint32_t val1 = l0_x + l1_x + l2_x + l3_x + l4_x + l5_x + l6_x + *old_value1;
			const uint32_t val2 = l0_y + l1_y + l2_y + l3_y + l4_y + l5_y + l6_y + *old_value2;
			const uint32_t val3 = l0_z + l1_z + l2_z + l3_z + l4_z + l5_z + l6_z + *old_value3;
			const uint32_t val4 = l0_w + l1_w + l2_w + l3_w + l4_w + l5_w + l6_w + *old_value4;
			const uint32_t val5 = l0_x2 + l1_x2 + l2_x2 + l3_x2 + l4_x2 + l5_x2 + l6_x2 + *old_value5;
			const uint32_t val6 = l0_y2 + l1_y2 + l2_y2 + l3_y2 + l4_y2 + l5_y2 + l6_y2 + *old_value6;
			const uint32_t val7 = l0_z2 + l1_z2 + l2_z2 + l3_z2 + l4_z2 + l5_z2 + l6_z2 + *old_value7;
			const uint32_t val8 = l0_w2 + l1_w2 + l2_w2 + l3_w2 + l4_w2 + l5_w2 + l6_w2 + *old_value8;
		#else
			const uint32_t val1 = l0_x + l1_x + l2_x + *old_value1;
			const uint32_t val2 = l0_y + l1_y + l2_y + *old_value2;
			const uint32_t val3 = l0_z + l1_z + l2_z + *old_value3;
			const uint32_t val4 = l0_w + l1_w + l2_w + *old_value4;
			const uint32_t val5 = l0_x2 + l1_x2 + l2_x2 + *old_value5;
			const uint32_t val6 = l0_y2 + l1_y2 + l2_y2 + *old_value6;
			const uint32_t val7 = l0_z2 + l1_z2 + l2_z2 + *old_value7;
			const uint32_t val8 = l0_w2 + l1_w2 + l2_w2 + *old_value8;
		#endif
		int min_idx1 = dis;
		uint32_t min1 = val1;
		if(val1 > val2) {
			min1 = val2;
			min_idx1 = dis+1;
		}

		int min_idx2 = dis+2;
		uint32_t min2 = val3;
		if(val3 > val4) {
			min2 = val4;
			min_idx2 = dis+3;
		}

		// combine 1
		uint32_t minval1 = min1;
		int min_idx_1 = min_idx1;
		if (min1 > min2){
			minval1 = min2;
			min_idx_1 = min_idx2;
		}

		int min_idx3 = dis+4;
		uint32_t min3 = val5;
		if(val5 > val6) {
			min3 = val6;
			min_idx3 = dis+5;
		}

		int min_idx4 = dis+6;
		uint32_t min4 = val7;
		if(val7 > val8) {
			min4 = val8;
			min_idx4 = dis+7;
		}

		// combine 2
		uint32_t minval2 = min3;
		int min_idx_2 = min_idx3;
		if (min3 > min4){
			minval2 = min4;
			min_idx_2 = min_idx4;
		}
		
		uint32_t minval = minval1;
		int min_idx = min_idx_1;
		if(minval1 > minval2) {
			minval = minval2;
			min_idx = min_idx_2;
		}

		const int min_warpindex = warpReduceMinIndex(minval, min_idx);
		// return the last result
		if(lane == 0) {
			d_disparity[index_im] = min_warpindex;
		}
	} else {
		st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index]), *old_values);
		st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index + 4]), *old_values2);
	}

	// check for the min for 8 disparity
	if(min_type == MIN_COMPUTE) {
		int min_cost_scalar1 = min(min(*old_value1, *old_value2), min(*old_value3, *old_value4));
		int min_cost_scalar2 = min(min(*old_value5, *old_value6), min(*old_value7, *old_value8));
		int min_cost_scalar = min(min_cost_scalar1, min_cost_scalar2);
		*min_cost = uchar_to_uint32(warpReduceMin(min_cost_scalar));
		*min_cost_p2 = *min_cost + p2_vector;
	}
}

// the difference between CostAggregationDiagonalGeneric
// use kernl time different
template<class T, int add_col, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int cols, int add_index, const T *_d_transform0, const T *_d_transform1, const int add_imindex, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int lane = threadIdx.x % WARP_SIZE;
	// 128/32=4
	const int dis = 8 * lane;
	int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
	int col, index_im;
	if(recompute || join_dispcomputation) {
		if(recompute) {
			col = initial_col;
		}
		index_im = initial_row*cols+initial_col;
	}

	const int MAX_PAD = UCHAR_MAX-P1;
	const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
	const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
	// important cost aggrestion
	// batch 8
	int old_value1 = 0;
	int old_value2 = 0;
	int old_value3 = 0;
	int old_value4 = 0;
	int old_value5 = 0;
	int old_value6 = 0;
	int old_value7 = 0;
	int old_value8 = 0;

	uint32_t min_cost, min_cost_p2;
	uint32_t old_values = 0;
	uint32_t old_values2 = 0;

	T rp0 = 0; 
	T rp1 = 0;
	T rp2 = 0;
	T rp3 = 0;
	T rp4 = 0;
	T rp5 = 0;
	T rp6 = 0;
	T rp7 = 0;

	// recompute first
	if(recompute) {
		if(dir_type == DIR_LEFTRIGHT) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			for(int i = 8; i < max_iter-8; i+=8) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3,  &rp4, &rp5,d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		} else if(dir_type == DIR_RIGHTLEFT) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7,  &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			// why use 4 as a batch?
			for(int i = 8; i < max_iter-7; i+=8) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3,&rp4, &rp5, &rp6, &rp7, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7,  &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp4, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2,  &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp5, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp6, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5,  d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp7, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		} else {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			for(int i = 1; i < max_iter; i++) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		}
	} else {
		CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

		for(int i = 1; i < max_iter; i++) {
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		}
		CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
		CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

// this is the kernl for the iteration
// different from CostAggregationGeneric, Diagonal
template<int add_index, class T, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationDiagonalGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int col_nomin, const int col_copycost, const int cols, const T *_d_transform0, const T *_d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int lane = threadIdx.x % WARP_SIZE;
	// 128/32=4
	const int dis = 8*lane;
	int col = initial_col;
	int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
	int index_im;
	if(recompute || join_dispcomputation) {
		index_im = initial_row*cols+col;
	}
	const int MAX_PAD = UCHAR_MAX-P1;
	// combine to uchar4
	const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
	const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
	int old_value1 = 0;
	int old_value2 = 0;
	int old_value3 = 0;
	int old_value4 = 0;
	int old_value5 = 0;
	int old_value6 = 0;
	int old_value7 = 0;
	int old_value8 = 0;

	uint32_t min_cost, min_cost_p2;

	uint32_t old_values = 0;
	uint32_t old_values2 = 0;
	
	T rp0 = 0; 
	T rp1 = 0;
	T rp2 = 0;
	T rp3 = 0;
	T rp4 = 0;
	T rp5 = 0;
	T rp6 = 0;
	T rp7 = 0;

	CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	// no batch ?
	for(int i = 1; i < max_iter; i++) {
		CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, initial_row, i, dis);
		if(col == col_copycost) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		} else {
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
		}
	}

	CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, max_iter, initial_row, dis);
	if(col == col_copycost) {
		CostAggregationGenericIteration<T, ITER_COPY, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	} else {
		CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_values2, &old_value1, &old_value2, &old_value3, &old_value4, &old_value5, &old_value6, &old_value7, &old_value8, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, &rp4, &rp5, &rp6, &rp7, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalDownUpRightLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	// refer https://forums.developer.nvidia.com/t/difference-between-threadidx-blockidx-statements/12161/2
	// int idx = blockDim.x*blockIdx.x + threadIdx.x
	const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
	if(initial_col < cols) {
		const int initial_row = rows-1;
		const int add_index = -1;
		const int col_nomin = 0;
		const int col_copycost = cols-1;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>
__global__ void CostAggregationKernelDiagonalDownUpLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
	if(initial_col >= 0) {
		const int initial_row = rows-1;
		const int add_index = 1;
		const int col_nomin = cols-1;
		const int col_copycost = 0;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalUpDownRightLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = -1;
		const int col_nomin = 0;
		const int col_copycost = cols-1;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = PATH_AGGREGATION == 8;

		CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalUpDownLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = 1;
		const int col_nomin = cols-1;
		const int col_copycost = 0;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>

__global__ void CostAggregationKernelLeftToRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_row < rows) {
		const int initial_col = 0;
		const int add_index = MAX_DISPARITY;
		const int add_imindex = 1;
		const int max_iter = cols-1;
		const int add_col = 1;
		//const bool recompute = true;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_LEFTRIGHT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>

__global__ void CostAggregationKernelRightToLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_row < rows) {
		const int initial_col = cols-1;
		const int add_index = -MAX_DISPARITY;
		const int add_imindex = -1;
		const int max_iter = cols-1;
		const int add_col = -1;
		//const bool recompute = true;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>
__global__ void CostAggregationKernelDownToUp(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = rows-1;
		const int add_index = -cols*MAX_DISPARITY;
		const int add_imindex = -cols;
		const int max_iter = rows-1;
		const int add_col = 0;
		const bool recompute = false;
		const bool join_dispcomputation = PATH_AGGREGATION == 4;

		CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

template<class T>
//__launch_bounds__(64, 16)
__global__ void CostAggregationKernelUpToDown(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t * __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = cols*MAX_DISPARITY;
		const int add_imindex = cols;
		const int max_iter = rows-1;
		const int add_col = 0;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	}
}

#endif /* COST_AGGREGATION_H_ */
