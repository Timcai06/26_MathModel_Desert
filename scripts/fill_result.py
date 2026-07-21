from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", MAIN_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("x14ac", "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac")


def _set_numeric(root: ET.Element, reference: str, value: int | float) -> None:
    cell = root.find(f".//{{{MAIN_NS}}}c[@r='{reference}']")
    if cell is None:
        raise ValueError(f"模板缺少单元格 {reference}")
    cell.attrib.pop("t", None)
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None:
        value_node = ET.SubElement(cell, f"{{{MAIN_NS}}}v")
    value_node.text = str(value)


def _records_by_day(item: dict) -> dict[int, dict]:
    records = {record["day"]: record for record in item["records"]}
    arrival_day = item["arrival_day"]
    terminal = records[arrival_day]
    for day in range(arrival_day + 1, 31):
        records[day] = {**terminal, "day": day}
    return records


def fill(template: Path, solution_json: Path, output: Path) -> None:
    payload = json.loads(solution_json.read_text(encoding="utf-8"))
    by_level = {item["level"]: item for item in payload}
    if set(by_level) != {1, 2}:
        raise ValueError("精确解 JSON 必须同时包含第一关和第二关")

    with ZipFile(template, "r") as source:
        sheet_xml = source.read("xl/worksheets/sheet1.xml")
        root = ET.fromstring(sheet_xml)
        for level, columns in ((1, ("B", "C", "D", "E")), (2, ("H", "I", "J", "K"))):
            records = _records_by_day(by_level[level])
            for day in range(31):
                row = day + 4
                record = records[day]
                values = (record["location"], record["cash"], record["water"], record["food"])
                for column, value in zip(columns, values, strict=True):
                    _set_numeric(root, f"{column}{row}", value)

        updated_sheet = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as target:
            for info in source.infolist():
                data = updated_sheet if info.filename == "xl/worksheets/sheet1.xml" else source.read(info.filename)
                target.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="把精确求解结果写入竞赛 Result.xlsx 模板")
    parser.add_argument("template", type=Path)
    parser.add_argument("solution_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fill(args.template, args.solution_json, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
