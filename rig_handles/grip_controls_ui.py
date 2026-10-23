"""Grip controls: 3D view sidebar (N) > Grip tab.

Named sliders for every grip rig handle, resets, mock-up show/export. Lives in the .blend as the text
block 'grip_controls_ui.py' (registers on load when the file is trusted; otherwise open it in the
Text Editor and press Run Script).
"""
import json
import os

import bpy

RIG = "grip rig"
MOCK_COLL = "mock-up print"
CASE_COLLS = ("case", "parts")

# (bone, title, [(kind, index, label), ...])  kind: loc = pose location (mm), prop = custom property
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
    ("R2 button top", "R2 trigger", [("loc", 0, "X  sideways"), ("loc", 1, "Y  toward bumpers"),
                                      ("loc", 2, "Z  toward face"), ("prop", "arch radius", "arch radius")]),
    ("R4 button top", "R4 trigger", [("loc", 0, "X  sideways"), ("loc", 1, "Y  toward bumpers"),
                                      ("loc", 2, "Z  toward face"), ("prop", "arch radius", "arch radius")]),
]


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


class GRIP_OT_reset_scan(bpy.types.Operator):
    bl_idname = "grip.reset_scan"
    bl_label = "Back to scan result"
    bl_description = "Put every handle back where the scan + solver put it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rig = rig_obj()
        apply_pose(rig, json.loads(rig["scan_pose"]))
        return {"FINISHED"}


class GRIP_OT_reset_original(bpy.types.Operator):
    bl_idname = "grip.reset_original"
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
        rig.update_tag()
        return {"FINISHED"}


class GRIP_OT_reset_one(bpy.types.Operator):
    bl_idname = "grip.reset_one"
    bl_label = "Reset"
    bl_description = "Put this handle back to the scan result"
    bl_options = {"REGISTER", "UNDO"}
    bone: bpy.props.StringProperty()

    def execute(self, context):
        rig = rig_obj()
        P = json.loads(rig["scan_pose"])
        apply_pose(rig, {self.bone: P[self.bone]})
        return {"FINISHED"}


class GRIP_OT_select(bpy.types.Operator):
    bl_idname = "grip.select"
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


class GRIP_OT_export_mockup(bpy.types.Operator):
    bl_idname = "grip.export_mockup"
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
        op = head.operator("grip.select", text="", icon="RESTRICT_SELECT_OFF")
        op.bone = bone
        op = head.operator("grip.reset_one", text="", icon="LOOP_BACK")
        op.bone = bone
        col = box.column(align=True)
        for kind, idx, label in rows:
            if kind == "loc":
                col.prop(pb, "location", index=idx, text=label + " (mm)")
            else:
                col.prop(pb, f'["{idx}"]', text=label + " (mm)")


class GRIP_PT_controls(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip"
    bl_label = "Grip controls"

    @classmethod
    def poll(cls, context):
        return rig_obj() is not None

    def draw(self, context):
        rig = rig_obj()
        lay = self.layout
        lay.label(text="Drag a ball with G, or use the sliders.")
        row = lay.row(align=True)
        row.operator("grip.reset_scan", icon="LOOP_BACK")
        row.operator("grip.reset_original", icon="CANCEL")


class GRIP_PT_shape(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip"
    bl_label = "Grip shape"
    bl_parent_id = "GRIP_PT_controls"

    def draw(self, context):
        draw_bones(self.layout, rig_obj(), GRIP)


class GRIP_PT_buttons(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip"
    bl_label = "R2 / R4 triggers"
    bl_parent_id = "GRIP_PT_controls"

    def draw(self, context):
        self.layout.label(text="Ball = centre of the button's press face")
        draw_bones(self.layout, rig_obj(), BUTTONS)


class GRIP_PT_mockup(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Grip"
    bl_label = "Mock-up print"
    bl_parent_id = "GRIP_PT_controls"

    def draw(self, context):
        lay = self.layout
        c = _coll_toggle(MOCK_COLL)
        if c is not None:
            lay.prop(c, "hide_viewport", text="Hide mock-up (faster posing)", toggle=True)
        for n in CASE_COLLS:
            c = _coll_toggle(n)
            if c is not None:
                lay.prop(c, "hide_viewport", text=f"Hide '{n}' parts (faster posing)", toggle=True)
        lay.operator("grip.export_mockup", icon="EXPORT")


CLASSES = (GRIP_OT_reset_scan, GRIP_OT_reset_original, GRIP_OT_reset_one, GRIP_OT_select,
           GRIP_OT_export_mockup, GRIP_PT_controls, GRIP_PT_shape, GRIP_PT_buttons, GRIP_PT_mockup)


def register():
    for c in CLASSES:
        try:
            bpy.utils.unregister_class(getattr(bpy.types, c.__name__))
        except Exception:
            pass
        bpy.utils.register_class(c)


register()
