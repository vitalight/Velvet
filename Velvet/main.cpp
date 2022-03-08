#include <iostream>

#include "VtGraphics.h"

using namespace Velvet;

int main()
{
	VtGraphics graphics;
	graphics.Initialize();

	graphics.AddActor(VtActor::FixedTriangle());

	return graphics.Run();
}