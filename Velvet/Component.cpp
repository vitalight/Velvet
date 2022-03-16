#include "Component.hpp"
#include "Actor.hpp"

using namespace std;
using namespace Velvet;

shared_ptr<Transform> Component::transform()
{
	if (actor)
	{
		return actor->transform;
	}
	else
	{
		return make_shared<Transform>(Transform(nullptr));
	}
}
