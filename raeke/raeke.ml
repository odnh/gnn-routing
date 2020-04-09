open Core

open Yates_routing.Util
open Yates_types.Types

let get_input_topology path = Net.Parse.from_dotfile path

let get_dummy_data topo = (get_hosts topo, SrcDstMap.empty)

let raeke_routing dot_file_path =
  let topo = get_input_topology dot_file_path in
  let (_, demands) = get_dummy_data topo in
  Yates_routing.Raeke.initialize SrcDstMap.empty;
  let scheme = Yates_routing.Raeke.solve topo demands in
  let oc = Out_channel.stdout in
  fprintf oc "%s\n" (Yates_routing.Util.dump_scheme topo scheme)

let _ =
  let path = Sys.argv.(1) in
  raeke_routing path
