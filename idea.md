I want to make mechanical-looking things
I want to draw them with wire frames ... ? Maybe.. Maybe just use tris for starters

I want the machines to appear to be interacting, like gears and stuff

Or maybe I want to do a particle-routing-like thing. So that the particles move along paths determined by mixing their vertices?

Hm

I would maybe need different code for that, and it's okay.


Make a design, favoring right angles
Then repeat it a few times along a dimension with a random step
and then mutate it

Or maybe only pick a handfull of designs and mutate them

The genes might be:
* dim A is one of XYZ
* dim B is one of XYZ excluding A
* Number of dim A steps
* Number of dim B steps
* A increment
* B increment

Ooh or maybe a recursive design;
Regular settings, and then also instructions to loop the last x instructions y times
Be very careful not to make a turing machine (or any other computing machine for that matter)

Mini-programs defined in bytes so that any random byte array is a valid program
* Move n units
* Switch direction to one of XYZ or one of -XYZ
* Jump point (see below)
* Jump to the last jump point (or beginning) n times and then pass through this statement (total n multiplied by all inner loops should be capped) - or we should just cap the number of steps. Let's do that instead. N should be low.
* Cancel next plot

So the idea is to mutate a random byte in each program and run it again
