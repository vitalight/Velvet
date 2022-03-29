#pragma once

#include <functional>

#include "Material.hpp"

namespace Velvet
{
	using namespace std;

	// This class allows objects with same material to have different properties
	struct MaterialProperty
	{
		function<void(Material*)> preRendering;
	};
}