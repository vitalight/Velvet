#include "VtClothSolverGPU.cuh"
#include "Common.hpp"
#include "Common.cuh"
#include "Timer.hpp"

using namespace std;

namespace Velvet
{
	__device__ __constant__ VtSimParams d_params;
	VtSimParams h_params;

	void SetSimulationParams(VtSimParams* hostParams)
	{
		ScopedTimerGPU timer("Solver_SetParams");
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, hostParams, sizeof(VtSimParams)));
		h_params = *hostParams;
	}

	__global__ void InitializePositions_Impl(glm::vec3* positions, const int count, const glm::mat4 modelMatrix)
	{
		GET_CUDA_ID(id, count);
		positions[id] = modelMatrix * glm::vec4(positions[id], 1);
	}

	void InitializePositions(glm::vec3* positions, int count, glm::mat4 modelMatrix)
	{
		ScopedTimerGPU timer("Solver_Initialize");
		CUDA_CALL(InitializePositions_Impl, count)(positions, count, modelMatrix);
	}

	__global__ void PredictPositions_Impl(CONST(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		glm::vec3 gravity = glm::vec3(0, -10, 0);
		velocities[id] += d_params.gravity * deltaTime;
		predicted[id] = positions[id] + velocities[id] * deltaTime;
	}

	void PredictPositions(CONST(glm::vec3*) positions, glm::vec3* predicted, glm::vec3* velocities, float deltaTime)
	{
		ScopedTimerGPU timer("Solver_Predict");
		CUDA_CALL(PredictPositions_Impl, h_params.numParticles)(positions, predicted, velocities, deltaTime);
	}

	__device__ void AtomicAdd(glm::vec3* address, int index, glm::vec3 val, int reorder)
	{
		int r1 = reorder % 3;
		int r2 = (reorder+1) % 3;
		int r3 = (reorder+2) % 3;
		atomicAdd(&(address[index].x)+r1, val[r1]);
		atomicAdd(&(address[index].x)+r2, val[r2]);
		atomicAdd(&(address[index].x)+r3, val[r3]);
	}

	__global__ void SolveStretch_Impl(uint numConstraints, CONST(int*) stretchIndices, CONST(float*) stretchLengths, 
		CONST(float*) inverseMass, CONST(glm::vec3*) predicted, glm::vec3* positionDeltas, int* positionDeltaCount)
	{
		GET_CUDA_ID(id, numConstraints);

		int idx1 = stretchIndices[2 * id];
		int idx2 = stretchIndices[2 * id + 1];
		float expectedDistance = stretchLengths[id];

		glm::vec3 diff = predicted[idx1] - predicted[idx2];
		float distance = glm::length(diff);
		float w1 = inverseMass[idx1];
		float w2 = inverseMass[idx2];

		if (distance != expectedDistance && w1 + w2 > 0)
		{
			glm::vec3 gradient = diff / (distance + EPSILON);
			// compliance is zero, therefore XPBD=PBD
			float denom = w1 + w2;
			float lambda = (distance - expectedDistance) / denom;
			glm::vec3 common = lambda * gradient;
			glm::vec3 correction1 = -w1 * common;
			glm::vec3 correction2 = w2 * common;
			int reorder = idx1 + idx2;
			AtomicAdd(positionDeltas, idx1, correction1, reorder);
			AtomicAdd(positionDeltas, idx2, correction2, reorder);
			atomicAdd(&positionDeltaCount[idx1], 1);
			atomicAdd(&positionDeltaCount[idx2], 1);
			//printf("correction[%d] = (%.2f,%.2f,%.2f)\n", idx1, correction1.x, correction1.y, correction1.z);
			//printf("correction[%d] = (%.2f,%.2f,%.2f)\n", idx2, correction2.x, correction2.y, correction2.z);
		}
	}


	void SolveStretch(uint numConstraints, CONST(int*) stretchIndices, CONST(float*) stretchLengths,
		CONST(float*) inverseMass, glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount)
	{
		ScopedTimerGPU timer("Solver_SolveStretch");
		CUDA_CALL(SolveStretch_Impl, numConstraints)(numConstraints, stretchIndices, stretchLengths, inverseMass, predicted, positionDeltas, positionDeltaCount);
	}

	__global__ void SolveBending_Impl(
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		uint numConstraints,
		float deltaTime)
	{
		GET_CUDA_ID(id, numConstraints);
		uint idx1 = bendingIndices[id * 4];
		uint idx2 = bendingIndices[id * 4+1];
		uint idx3 = bendingIndices[id * 4+2];
		uint idx4 = bendingIndices[id * 4+3];
		float expectedAngle = bendingAngles[id];

		float w1 = invMass[idx1];
		float w2 = invMass[idx2];
		float w3 = invMass[idx3];
		float w4 = invMass[idx4];

		glm::vec3 p1 = predicted[idx1];
		glm::vec3 p2 = predicted[idx2] - p1;
		glm::vec3 p3 = predicted[idx3] - p1;
		glm::vec3 p4 = predicted[idx4] - p1;
		glm::vec3 n1 = glm::normalize(glm::cross(p2, p3));
		glm::vec3 n2 = glm::normalize(glm::cross(p2, p4));

		float d = clamp(glm::dot(n1, n2), 0.0f, 1.0f);
		float angle = acos(d);
		// cross product for two equal vector produces NAN
		if (angle < EPSILON || isnan(d)) return;

		glm::vec3 q3 = (glm::cross(p2, n2) + glm::cross(n1, p2) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON);
		glm::vec3 q4 = (glm::cross(p2, n1) + glm::cross(n2, p2) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
		glm::vec3 q2 = -(glm::cross(p3, n2) + glm::cross(n1, p3) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON)
			- (glm::cross(p4, n1) + glm::cross(n2, p4) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
		glm::vec3 q1 = -q2 - q3 - q4;

		float xpbd_bend = d_params.bendCompliance / deltaTime / deltaTime;
		float denom = xpbd_bend + (w1 * glm::dot(q1, q1) + w2 * glm::dot(q2, q2) + w3 * glm::dot(q3, q3) + w4 * glm::dot(q4, q4));
		if (denom < EPSILON) return; // ?
		float lambda = sqrt(1.0f - d * d) * (angle - expectedAngle) / denom;

		int reorder = idx1 + idx2 + idx3 + idx4;
		AtomicAdd(positionDeltas, idx1, w1 * lambda * q1, reorder);
		AtomicAdd(positionDeltas, idx2, w2 * lambda * q2, reorder);
		AtomicAdd(positionDeltas, idx3, w3 * lambda * q3, reorder);
		AtomicAdd(positionDeltas, idx4, w4 * lambda * q4, reorder);
		
		atomicAdd(&positionDeltaCount[idx1], 1);
		atomicAdd(&positionDeltaCount[idx2], 1);
		atomicAdd(&positionDeltaCount[idx3], 1);
		atomicAdd(&positionDeltaCount[idx4], 1);
	}

	void SolveBending(
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		uint numConstraints,
		float deltaTime)
	{
		ScopedTimerGPU timer("Solver_SolveBending");
		CUDA_CALL(SolveBending_Impl, numConstraints)(predicted, positionDeltas, positionDeltaCount, bendingIndices, bendingAngles, invMass, numConstraints, deltaTime);
	}

	__global__ void SolveAttachment_Impl(
		int numConstraints,
		CONST(float*) invMass,
		CONST(int*) attachIndices, 
		CONST(glm::vec3*) attachPositions, 
		CONST(float*) attachDistances,
		CONST(glm::vec3*) predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount)
	{
		GET_CUDA_ID(id, numConstraints);

		uint pid = attachIndices[id];

		glm::vec3 attachPoint = attachPositions[id];
		float targetDist = attachDistances[id] * d_params.longRangeStretchiness;
		if (invMass[pid] == 0 && targetDist > 0) return;

		glm::vec3 pred = predicted[pid];
		glm::vec3 diff = pred - attachPoint;
		float dist = glm::length(diff);

		if (dist > targetDist)
		{
			//float coefficient = max(targetDist, dist - 0.1*d_params.particleDiameter);// 0.05 * targetDist + 0.95 * dist;
			glm::vec3 correction = -diff + diff / dist * targetDist;
			AtomicAdd(positionDeltas, pid, correction, id);
			atomicAdd(&positionDeltaCount[pid], 1);
		}
	}

	void SolveAttachment(
		int numConstraints,
		CONST(float*) invMass,
		CONST(int*) attachIndices,
		CONST(glm::vec3*) attachPositions,
		CONST(float*) attachDistances,
		glm::vec3* predicted,
		glm::vec3* positionDeltas,
		int* positionDeltaCount)
	{
		ScopedTimerGPU timer("Solver_SolveAttach");
		CUDA_CALL(SolveAttachment_Impl, numConstraints)(numConstraints, invMass, attachIndices, attachPositions, attachDistances, predicted, positionDeltas, positionDeltaCount);
	}

	__global__ void ApplyDeltas_Impl(glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		float count = (float)positionDeltaCount[id];
		if (count > 0)
		{
			predicted[id] += positionDeltas[id] / count * d_params.relaxationFactor;
			positionDeltas[id] = glm::vec3(0);
			positionDeltaCount[id] = 0;
		}
	}

	void ApplyDeltas(glm::vec3* predicted, glm::vec3* positionDeltas, int* positionDeltaCount)
	{
		ScopedTimerGPU timer("Solver_ApplyDeltas");
		CUDA_CALL(ApplyDeltas_Impl, h_params.numParticles)(predicted, positionDeltas, positionDeltaCount);
	}

	__device__ glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relVel)
	{
		glm::vec3 friction = glm::vec3(0);
		float correctionLength = glm::length(correction);
		if (d_params.friction > 0 && correctionLength > 0)
		{
			glm::vec3 norm = correction / correctionLength;

			glm::vec3 tanVel = relVel - norm * glm::dot(relVel, norm);
			float tanLength = glm::length(tanVel);
			float maxTanLength = correctionLength * d_params.friction;

			friction = -tanVel * min(maxTanLength / tanLength, 1.0f);
		}
		return friction;
	}

	__global__ void CollideSDF_Impl(const uint numColliders, CONST(SDFCollider*) colliders, CONST(glm::vec3*) positions, glm::vec3* predicted)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		auto pos = positions[id];
		auto pred = predicted[id];
		for (int i = 0; i < numColliders; i++)
		{
			auto collider = colliders[i];
			glm::vec3 correction = collider.ComputeSDF(pred, d_params.collisionMargin);
			pred += correction;

			glm::vec3 relVel = pred - pos;
			auto friction = ComputeFriction(correction, relVel);
			pred += friction;
		}
		predicted[id] = pred;
	}

	void CollideSDF(const uint numColliders, CONST(SDFCollider*) colliders, CONST(glm::vec3*) positions, glm::vec3* predicted)
	{
		ScopedTimerGPU timer("Solver_CollideSDFs");
		if (numColliders == 0) return;
		
		CUDA_CALL(CollideSDF_Impl, h_params.numParticles)(numColliders, colliders, positions, predicted);
	}

	__global__ void CollideParticles_Impl(
		CONST(float*) inverseMass,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions,
		glm::vec3* predicted)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		glm::vec3 positionDelta = glm::vec3(0);
		int deltaCount = 0;
		glm::vec3 pred_i = predicted[id];
		glm::vec3 vel_i = (pred_i - positions[id]);
		float w_i = inverseMass[id];

		for (int neighbor = id * d_params.maxNumNeighbors; neighbor < (id + 1) * d_params.maxNumNeighbors; neighbor++)
		{
			uint j = neighbors[neighbor];
			if (j > d_params.numParticles) break;
			//if (j > id) continue;

			float expectedDistance = d_params.particleDiameter;

			glm::vec3 pred_j = predicted[j];
			glm::vec3 diff = pred_i - pred_j;
			float distance = glm::length(diff);
			float w_j = inverseMass[j];

			if (distance < expectedDistance && w_i + w_j > 0)
			{
				glm::vec3 gradient = diff / (distance + EPSILON);
				float denom = w_i + w_j;
				float lambda = (distance - expectedDistance) / denom;
				glm::vec3 common = lambda * gradient;

				positionDelta -= w_i * common;

				glm::vec3 relativeVelocity = vel_i - (pred_j - positions[j]);
				glm::vec3 friction = ComputeFriction(common, relativeVelocity);
				positionDelta += w_i * friction;
			}
		}

		predicted[id] += positionDelta;
	}

	void CollideParticles(
		CONST(float*) inverseMass,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions,
		glm::vec3* predicted)
	{
		ScopedTimerGPU timer("Solver_CollideParticles");
		CUDA_CALL(CollideParticles_Impl, h_params.numParticles)(inverseMass, neighbors, positions, predicted);
	}

	__global__ void Finalize_Impl(CONST(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		glm::vec3 raw_vel = (predicted[id] - positions[id]) / deltaTime;
		float raw_vel_len = glm::length(raw_vel);
		glm::vec3 new_pos = predicted[id];
		if (raw_vel_len > d_params.maxSpeed)
		{
			raw_vel = raw_vel / raw_vel_len * d_params.maxSpeed;
			new_pos = positions[id] + raw_vel * deltaTime;
			//printf("Limit vel[%.3f>%.3f] for id[%d]. Pred[%.3f,%.3f,%.3f], Pos[%.3f,%.3f,%.3f]\n", raw_vel_len, d_params.maxSpeed, id);
		}
		velocities[id] = raw_vel * (1 - d_params.damping * deltaTime);
		positions[id] = new_pos;
	}

	void Finalize(CONST(glm::vec3*) predicted, glm::vec3* velocities, glm::vec3* positions, float deltaTime)
	{
		ScopedTimerGPU timer("Solver_Finalize");
		CUDA_CALL(Finalize_Impl, h_params.numParticles)(predicted, velocities, positions, deltaTime);
	}

	__global__ void ComputeTriangleNormals(uint numTriangles, CONST(glm::vec3*) positions, CONST(uint*) indices, glm::vec3* normals)
	{
		GET_CUDA_ID(id, numTriangles);
		uint idx1 = indices[id * 3];
		uint idx2 = indices[id * 3+1];
		uint idx3 = indices[id * 3+2];

		auto p1 = positions[idx1];
		auto p2 = positions[idx2];
		auto p3 = positions[idx3];

		auto normal = glm::cross(p2 - p1, p3 - p1);
		int reorder = idx1 + idx2 + idx3;
		AtomicAdd(normals, idx1, normal, reorder);
		AtomicAdd(normals, idx2, normal, reorder);
		AtomicAdd(normals, idx3, normal, reorder);
	}

	__global__ void ComputeVertexNormals(glm::vec3* normals)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		//normals[id] = glm::vec3(0,1,0);
		normals[id] = glm::normalize(normals[id]);
	}

	void ComputeNormal(uint numTriangles, CONST(glm::vec3*) positions, CONST(uint*) indices, glm::vec3* normals)
	{
		ScopedTimerGPU timer("Solver_UpdateNormals");
		if (h_params.numParticles)
		{
			cudaMemsetAsync(normals, 0, h_params.numParticles * sizeof(glm::vec3));
			CUDA_CALL(ComputeTriangleNormals, numTriangles)(numTriangles, positions, indices, normals);
			CUDA_CALL(ComputeVertexNormals, h_params.numParticles)(normals);
		}
	}

}