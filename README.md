# Mutautomata
This is a pattern generator based on a random set of instructions, which are manipulated in small ways and drawn several times. The plots this program makes are three-dimensional, and can optionally be displayed for VR. Without any arguments, a random seed is picked. 

```
cargo run --release -- -s 9359628770
```
![Example](example.png)


```
cargo run --release -- -s 10210725752218678800
```
![Example](example2.png)


```sh
cargo run --release -- --help
```

```
MutAutomata 0.1.0
Builder of spindly structures and such

USAGE:
    mutautomata [FLAGS] [OPTIONS]

FLAGS:
    -h, --help                 Prints help information
        --show-instructions    Show instructions of the seed
    -V, --version              Prints version information
        --vr                   Enable VR view

OPTIONS:
    -g, --genome-length <genome-length>                  Size of the genome [default: 8000]
        --max-bright <max-bright>                        Maximum brightness of colors [default: 1.0]
        --max-colors <max-colors>                        Maximum number of colors [default: 10]
    -u, --max-mutations <max-mutations>                  Maximum number of mutations per child [default: 100]
    -m, --max-steps-per-object <max-steps-per-object>
            Max steps which can be taken for rendering each shape [default: 300000]

        --min-bright <min-bright>                        Minimum brightness of colors [default: 0.0]
    -p, --plot-mode <plot-mode>                          Plot using this primitive/mode [default: lines]
    -s, --seed <seed>                                    Supply a seed
    -y, --str-seed <str-seed>                            Seed, as a string
    -v, --vertex-budget <vertex-budget>                  Max vertices to display [default: 3000000]
```
