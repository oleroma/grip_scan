"""Grip controls: 3D view sidebar (N) > Grip tab.

Named sliders for every grip rig handle, resets, mock-up show/export. Lives in the .blend as the text
block 'grip_controls_ui.py' (registers on load when the file is trusted; otherwise open it in the
Text Editor and press Run Script).
"""
import json
import math
import os

import bpy

RIG = "grip rig"
MOCK_COLL = "mock-up print"
CASE_COLLS = ("case", "parts")

# (bone, title, [(kind, index, label), ...])  kind: loc = pose location (mm), prop = custom property
HAND = [
    ("R1 hand", "Hand on R1", [("loc", 0, "contact along the finger (+ = toward tip)"),
                               ("rot", 0, "turn about R1")]),
]
GRIP = [
    ("palm swell", "Palm swell", [("loc", 1, "out / in"), ("loc", 0, "along A"), ("loc", 2, "along B")]),
    ("fin middle handle", "Middle finger fin", [("loc", 1, "out / in"), ("loc", 0, "sideways")]),
    ("fin ring handle", "Ring finger fin", [("loc", 1, "out / in"), ("loc", 0, "sideways")]),
    ("fin pinky handle", "Pinky fin", [("loc", 1, "out / in"), ("loc", 0, "sideways")]),
    ("stand handle", "Handle end (stand)", [("loc", 1, "out / in"), ("loc", 0, "sideways")]),
    ("thenar surface", "Thumb base (thenar)", [("loc", 2, "out / in"), ("loc", 0, "along A"),
                                               ("loc", 1, "along B")]),
]
BUTTONS = [
    ("R2 button top", "R2 trigger", [("loc", 0, "across the face"), ("loc", 1, "out toward the finger"),
                                      ("loc", 2, "up toward the face buttons"), ("prop", "arch radius", "arch radius"),
                                      ("rot", 0, "tilt (tip angle offset)"), ("rot", 2, "sideways tilt (tip shear)")]),
    ("R4 button top", "R4 trigger", [("loc", 0, "across the face"), ("loc", 1, "out toward the finger"),
                                      ("loc", 2, "up toward the face buttons"), ("prop", "arch radius", "arch radius"),
                                      ("rot", 0, "tilt (tip angle offset)"), ("rot", 2, "sideways tilt (tip shear)")]),
]


def press_angle(part):
    """Angle (deg) between the press face normal and the direction the face travels when pressed
    (the arch tangent at the face). Same math as trigger_controls/trigger_model.py inverse()."""
    node = bpy.data.node_groups[f"GripV {part} parameters"].nodes["button top control"]
    g = lambda n: float(node.inputs[n].default_value)  # noqa: E731
    tb, h, bro, ph, xo = g("top bevel"), g("height"), g("button radius offset"), g("button pressed height"), g("x offset")
    o, sh, R = g("tip angle offset"), g("tip shear"), g("arch radius")
    smd = node.inputs["SMD button"].default_value.matrix_world.translation
    top = node.inputs["button top"].default_value
    d = [top[0] - smd[0], top[1] - smd[1], top[2] - (smd[2] - 1.6 - ph)]
    rho = R - h / 2
    a = 0.0
    for _ in range(8):
        v = (-sh, 1.0, -math.tan(a - o))
        ln = math.sqrt(sum(c * c for c in v))
        n = (v[0] / ln, (v[1] * math.cos(a) - v[2] * math.sin(a)) / ln, (v[1] * math.sin(a) + v[2] * math.cos(a)) / ln)
        qx = xo + tb * n[0]
        qy = math.sqrt(max(d[0] ** 2 + d[1] ** 2 - qx ** 2, 0.0))
        U = qy - tb * n[1] + bro
        V = d[2] - tb * n[2]
        W = max(math.hypot(U, V), 1e-6)
        xx = (R * R - rho * rho + W * W) / (2 * W)
        yy = math.sqrt(max(R * R - xx * xx, 0.0))
        cy = (xx * U - yy * V) / W
        cz = (xx * V + yy * U) / W
        a = math.atan2(U - cy, cz - V)
    # in the arch frame the travel direction is +Y and the normal is (-sh, 1, -tan(a - o))
    return math.degrees(math.acos(1.0 / math.sqrt(sh * sh + 1.0 + math.tan(a - o) ** 2)))


def rig_obj():
    return bpy.data.objects.get(RIG)


def apply_pose(rig, P):
    for pb in rig.pose.bones:
        e = P.get(pb.name)
        if e is None:
            continue
        pb.location = e["loc"]
        pb.rotation_euler = e["rot"]
        if "arch radius" in e and "arch radius" in pb:
            pb["arch radius"] = e["arch radius"]
    rig.update_tag()


class GRIPRIG_OT_reset_scan(bpy.types.Operator):
    bl_idname = "griprig.reset_scan"
    bl_label = "Back to scan result"
    bl_description = "Put every handle back where the scan + solver put it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rig = rig_obj()
        apply_pose(rig, json.loads(rig["scan_pose"]))
        return {"FINISHED"}


class GRIPRIG_OT_reset_original(bpy.types.Operator):
    bl_idname = "griprig.reset_original"
    bl_label = "Back to original GripV"
    bl_description = "Clear every move: the grip and the R2/R4 triggers become the original GripV"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rig = rig_obj()
        orig = json.loads(rig.get("original_buttons", "{}"))
        for pb in rig.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
        for part, v in orig.items():
            pb = rig.pose.bones.get(part + " button top")
            if pb is not None:
                pb["arch radius"] = v["arch radius"]
                pb.rotation_euler = v.get("rot", (0.0, 0.0, 0.0))
        rig.update_tag()
        return {"FINISHED"}


class GRIPRIG_OT_reset_one(bpy.types.Operator):
    bl_idname = "griprig.reset_one"
    bl_label = "Reset"
    bl_description = "Put this handle back to the scan result"
    bl_options = {"REGISTER", "UNDO"}
    bone: bpy.props.StringProperty()

    def execute(self, context):
        rig = rig_obj()
        P = json.loads(rig["scan_pose"])
        apply_pose(rig, {self.bone: P[self.bone]})
        return {"FINISHED"}


class GRIPRIG_OT_select(bpy.types.Operator):
    bl_idname = "griprig.select"
    bl_label = "Select"
    bl_description = "Select this handle in the viewport (then G to drag it)"
    bone: bpy.props.StringProperty()

    def execute(self, context):
        rig = rig_obj()
        if context.mode != "POSE":
            for o in context.view_layer.objects:
                o.select_set(False)
            rig.hide_set(False)
            rig.select_set(True)
            context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode="POSE")
        for pb in rig.pose.bones:
            pb.select = False
        pb = rig.pose.bones[self.bone]
        pb.select = True
        rig.data.bones.active = pb.bone
        return {"FINISHED"}


class GRIPRIG_OT_export_mockup(bpy.types.Operator):
    bl_idname = "griprig.export_mockup"
    bl_label = "Export mock-up STLs"
    bl_description = "Write mockup_top.stl and mockup_bottom.stl into the folder 'mockup' next to this .blend"
    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        self.directory = bpy.path.abspath("//mockup/") if bpy.data.filepath else ""
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        coll = bpy.data.collections[MOCK_COLL]
        was_hidden = coll.hide_viewport
        coll.hide_viewport = False
        lc = _layer_coll(context.view_layer.layer_collection, MOCK_COLL)
        was_excluded = lc.exclude if lc else False
        if lc:
            lc.exclude = False
        os.makedirs(self.directory, exist_ok=True)
        prev_mode = context.mode
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        written = []
        for name in ("mock-up top", "mock-up bottom"):
            ob = bpy.data.objects[name]
            for o in context.view_layer.objects:
                o.select_set(False)
            ob.hide_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            path = os.path.join(self.directory, name.replace("mock-up ", "mockup_") + ".stl")
            bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, apply_modifiers=True)
            written.append(path)
        coll.hide_viewport = was_hidden
        if lc:
            lc.exclude = was_excluded
        self.report({"INFO"}, "wrote " + ", ".join(written))
        return {"FINISHED"}


def _layer_coll(lc, name):
    if lc.collection.name == name:
        return lc
    for c in lc.children:
        r = _layer_coll(c, name)
        if r:
            return r
    return None


def _coll_toggle(name):
    return bpy.data.collections.get(name)


def draw_bones(layout, rig, spec):
    for bone, title, rows in spec:
        pb = rig.pose.bones.get(bone)
        if pb is None:
            continue
        box = layout.box()
        head = box.row(align=True)
        head.label(text=title)
        op = head.operator("griprig.select", text="", icon="RESTRICT_SELECT_OFF")
        op.bone = bone
        op = head.operator("griprig.reset_one", text="", icon="LOOP_BACK")
        op.bone = bone
        col = box.column(align=True)
        for kind, idx, label in rows:
            if kind == "loc":
                col.prop(pb, "location", index=idx, text=label + " (mm)")
            elif kind == "rot":
                col.prop(pb, "rotation_euler", index=idx, text=label)
            else:
                col.prop(pb, f'["{idx}"]', text=label + " (mm)")


class GRIPRIG_PT_controls(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip rig"
    bl_label = "Grip controls"

    @classmethod
    def poll(cls, context):
        return rig_obj() is not None

    def draw(self, context):
        rig = rig_obj()
        lay = self.layout
        lay.label(text="Drag a ball with G, or use the sliders.")
        row = lay.row(align=True)
        row.operator("griprig.reset_scan", icon="LOOP_BACK")
        row.operator("griprig.reset_original", icon="CANCEL")


class GRIPRIG_PT_hand(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip rig"
    bl_label = "Hand on R1"
    bl_parent_id = "GRIPRIG_PT_controls"

    def draw(self, context):
        rig = rig_obj()
        pb = rig.pose.bones.get("R1 hand")
        info = json.loads(rig.get("index_curve", "{}"))
        if pb is not None and info:
            s = info["s0"] + pb.location[0]
            j = info["joints_mm"]
            part = ("distal" if s >= j["distal"] else "middle" if s >= j["middle"] else
                    "proximal" if s >= j["proximal"] else "knuckle")
            self.layout.label(text=f"R1 touches the index {info['length'] - s:.1f} mm from the tip ({part} phalanx)")
        self.layout.label(text="The hand slides along the finger; the palm swell follows the palm")
        draw_bones(self.layout, rig, HAND)
        pf = rig.pose.bones.get("palm follow")
        if pf is not None and "palm follow" in pf.constraints:
            self.layout.prop(pf.constraints["palm follow"], "influence", text="palm follow (1 = fully)")


class GRIPRIG_PT_shape(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip rig"
    bl_label = "Grip shape"
    bl_parent_id = "GRIPRIG_PT_controls"

    def draw(self, context):
        draw_bones(self.layout, rig_obj(), GRIP)


class GRIPRIG_PT_buttons(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip rig"
    bl_label = "R2 / R4 triggers"
    bl_parent_id = "GRIPRIG_PT_controls"

    def draw(self, context):
        self.layout.label(text="Handle = centre of the press face, arrow = where it faces")
        self.layout.label(text="G moves it, R rotates it (R X X / R Z Z for one axis)")
        rig = rig_obj()
        lim = float(rig.get("max press angle", 30.0))
        box = self.layout.box()
        box.label(text="Press angle = face vs the way the button moves")
        for part in ("R2", "R4"):
            try:
                ang = press_angle(part)
            except Exception:
                continue
            row = box.row()
            row.alert = ang > lim
            row.label(text=f"{part}: {ang:.0f} deg" + ("  TOO STEEP" if ang > lim else "  ok"),
                      icon="ERROR" if ang > lim else "CHECKMARK")
        box.prop(rig, '["max press angle"]', text="max press angle (deg)")
        draw_bones(self.layout, rig, BUTTONS)


class GRIPRIG_PT_mockup(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip rig"
    bl_label = "Mock-up print"
    bl_parent_id = "GRIPRIG_PT_controls"

    def draw(self, context):
        lay = self.layout
        c = _coll_toggle(MOCK_COLL)
        if c is not None:
            lay.prop(c, "hide_viewport", text="Hide mock-up (faster posing)", toggle=True)
        for n in CASE_COLLS:
            c = _coll_toggle(n)
            if c is not None:
                lay.prop(c, "hide_viewport", text=f"Hide '{n}' parts (faster posing)", toggle=True)
        lay.operator("griprig.export_mockup", icon="EXPORT")


CLASSES = (GRIPRIG_OT_reset_scan, GRIPRIG_OT_reset_original, GRIPRIG_OT_reset_one, GRIPRIG_OT_select,
           GRIPRIG_OT_export_mockup, GRIPRIG_PT_controls, GRIPRIG_PT_hand, GRIPRIG_PT_shape, GRIPRIG_PT_buttons, GRIPRIG_PT_mockup)


def register():
    for c in CLASSES:
        try:
            bpy.utils.unregister_class(getattr(bpy.types, c.__name__))
        except Exception:
            pass
        bpy.utils.register_class(c)
    bpy.app.timers.register(_open_gripv_workspace, first_interval=0.5)


def _open_gripv_workspace():
    # Background Blender cannot store the active workspace, so the file is saved on the GripX tab;
    # switch to the GripV tab once, the first time the file opens on the GripV scene.
    wm = bpy.context.window_manager
    ws = bpy.data.workspaces.get("GripV")
    if wm is None or ws is None or not wm.windows:
        return None
    for win in wm.windows:
        if win.scene.name == "GripV" and win.workspace is not ws:
            win.workspace = ws
    return None


register()
